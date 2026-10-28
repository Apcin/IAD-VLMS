"""No model weights, CUDA, or vLLM required: exercise the two-stage data flow."""

import copy
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
import dataset_json_store as store
import generate_traces_qwen3vl_evidence_first as generation
from compile_traces import compile_dataset
from test_pipeline import anomaly_sample


def evidence_value(view_id="query", visibility="clear"):
    return {
        "object_observation": "A round object has a smooth surface and a continuous outer rim.",
        "regional_observations": [{
            "view_id": view_id, "region": "upper rim",
            "query_evidence": ["The rim has a small inward contour change."],
            "normal_comparison": "The comparable reference contour appears more even; exact alignment is uncertain.",
            "visibility": visibility, "uncertainty": "Specular reflection limits fine texture comparison.",
        }],
    }


def evidence_record(sample):
    return {"image_id": sample["id"], "dataset": sample["dataset"],
            "evidence": evidence_value(sample["query_images"][0]["view_id"], "not_visible")}


def trace_value(sample):
    think = " ".join(["visible"] * 90)
    if not sample["is_anomaly"]:
        reasoning = f"<think>{think}</think><answer>No</answer>"
    else:
        location = json.dumps(generation.baseline.ground_truth_location_value(sample))
        reasoning = (f"<think>{think}</think><location>{location}</location>"
                     f"<type>{sample['supervision']['anomaly_type']['canonical']}</type><answer>Yes</answer>")
    return {"image_id": sample["id"], "reasoning": reasoning}


class ProcessorStub:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return messages[0]["content"] + "\n" + messages[1]["content"][-1]["text"]


class EngineStub:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, requests, params):
        self.requests.append(requests)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=next(self.responses))]) for _ in requests]


def concurrent_writer(path, worker, barrier):
    barrier.wait(timeout=15)
    for index in range(5):
        identity = f"{worker}_{index}"
        store.merge_records(Path(path), "test", {identity: {"image_id": identity, "dataset": "test"}})


class EvidenceFirstTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.configuration = patch.multiple(
            config, EVIDENCE_DIR=self.root / "evidence", EVIDENCE_FIRST_TRACE_DIR=self.root / "traces",
            EVIDENCE_FIRST_COMPILED_DIR=self.root / "compiled_new", TRACE_DIR=self.root / "baseline",
            COMPILED_DIR=self.root / "compiled_baseline", MANIFEST_DIR=self.root / "manifests",
            LOG_DIR=self.root / "logs",
        )
        self.configuration.start()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.configuration.stop)

    def prepared_sample(self, modality="2d", anomaly=True):
        sample = copy.deepcopy(anomaly_sample(modality))
        sample["dataset"] = "mvtec_ad" if modality == "2d" else "real3d"
        sample["id"] = f"{sample['dataset']}__test_crack_private_{anomaly}"
        sample["sample_key"] = "object/test/crack/private.png"
        sample["is_anomaly"] = anomaly
        sample["label"] = "Anomaly" if anomaly else "Normal"
        sample["question"] = "Are there any defects?"
        if not anomaly:
            sample["supervision"] = {"bbox_prompt_images": [], "mask_paths": [], "bboxes": [], "anomaly_type": None}
        for item in generation.baseline.teacher_image_items(sample):
            # Distinct existing paths; pixel loading is mocked separately.
            path = self.root / Path(item["path"]).name
            path.touch()
            for group in (sample["query_images"], sample["supervision"]["bbox_prompt_images"], sample["normal_reference"]["images"]):
                for entry in group:
                    if entry["path"] == item["path"]:
                        entry["path"] = str(path)
        return sample

    def test_stage1_allowlist_for_all_four_branches_and_retries(self):
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                sample = self.prepared_sample(modality, anomaly)
                processor = ProcessorStub()
                with patch.object(generation.baseline, "open_rgb", return_value=object()):
                    request, _ = generation.build_request(
                        sample, processor, config.STAGE1_PROMPT_PATH.read_text(), "evidence",
                        errors=[f"private crack path {sample['id']} [101,52,203,177]"],
                    )
                prompt = request["prompt"]
                for forbidden in (sample["id"], sample["sample_key"], "concavity", "[101,52,203,177]",
                                  "Ground Truth:", "Required response branch:", str(self.root)):
                    self.assertNotIn(forbidden, prompt)
                expected_count = (3 if anomaly else 2) * (1 if modality == "2d" else 2)
                self.assertEqual(len(request["multi_modal_data"]["image"]), expected_count)
                self.assertIn("view_id=" + sample["query_images"][0]["view_id"], prompt)
                self.assertNotIn("image_id", generation.VisualEvidence.model_json_schema()["properties"])

    def test_evidence_preserves_uncertainty_and_view_mapping(self):
        sample = self.prepared_sample("3d_multiview")
        value = evidence_value("view_000", "not_visible")
        self.assertEqual(generation.parse_evidence(json.dumps(value), sample).model_dump(), value)
        # view_000 has a cue image, but only view_017 has an actual GT box.
        text = generation.format_stage1_user_prompt(sample)
        self.assertIn("view_id=view_000", text)
        self.assertNotIn("view_017 is anomalous", text)
        value["regional_observations"][0]["view_id"] = "view_999"
        with self.assertRaises(ValueError):
            generation.parse_evidence(json.dumps(value), sample)

    def test_collection_concurrent_merges_and_atomic_failure(self):
        path = self.root / "evidence_test.json"
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(3)
        workers = [context.Process(target=concurrent_writer, args=(str(path), index, barrier)) for index in range(3)]
        for worker in workers:
            worker.start()
        try:
            for worker in workers:
                worker.join(timeout=25)
                self.assertEqual(worker.exitcode, 0)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
        self.assertEqual(len(store.read_records(path, "test")), 15)
        original = path.read_bytes()
        with patch.object(store.os, "replace", side_effect=OSError("simulated interrupted write")):
            with self.assertRaises(OSError):
                store.merge_records(path, "test", {"new": {"image_id": "new", "dataset": "test"}})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        with self.assertRaises(ValueError):
            store.read_records(path, "wrong_dataset")

    def test_stage1_retry_then_stage2_failure_reuses_disk_evidence(self):
        sample = self.prepared_sample()
        args = generation.parse_args(["--max-retries", "2"])
        processor = ProcessorStub()
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            engine = EngineStub(["invalid", json.dumps(evidence_value())])
            self.assertEqual(generation.process_chunk(
                [sample], engine, processor, None, "system", args, "evidence", {},
            ), (1, 0))
            evidence_path = generation.collection_path("evidence", sample["dataset"])
            before = evidence_path.read_bytes()
            records = store.read_records(evidence_path, sample["dataset"])
            self.assertEqual(records[sample["id"]]["attempt"], 2)
            self.assertEqual(generation.process_chunk(
                [sample], EngineStub(["bad", "bad"]), processor, None, "system", args, "cot", records,
            ), (0, 1))
            self.assertEqual(evidence_path.read_bytes(), before)
            self.assertEqual(generation.process_chunk(
                [sample], EngineStub([json.dumps(trace_value(sample))]), processor, None, "system", args, "cot",
                store.read_records(evidence_path, sample["dataset"]),
            ), (1, 0))
        trace_path = generation.collection_path("cot", sample["dataset"])
        self.assertTrue(generation.has_trace(sample, store.read_records(trace_path, sample["dataset"]), 0.9))
        self.assertEqual(list(config.EVIDENCE_FIRST_TRACE_DIR.glob("traces_*.json")), [trace_path])
        self.assertFalse(config.TRACE_DIR.exists())
        self.assertIn("The rim has a small inward contour change.", processor.messages[-1][1]["content"][-1]["text"])

    def test_stage1_exhausted_failure_does_not_create_cache(self):
        sample = self.prepared_sample()
        args = generation.parse_args(["--max-retries", "1"])
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            self.assertEqual(generation.process_chunk(
                [sample], EngineStub(["invalid"]), ProcessorStub(), None, "system", args, "evidence", {},
            ), (0, 1))
        self.assertFalse(generation.collection_path("evidence", sample["dataset"]).exists())

    def test_default_cli_resume_and_shards_without_loading_model(self):
        args = generation.parse_args([])
        self.assertEqual(args.datasets, ["mvtec_ad", "anomaly_shapenet", "mvtec3d", "real3d"])
        self.assertEqual(args.stage, "all")
        samples = [self.prepared_sample(anomaly=False) for _ in range(4)]
        for index, sample in enumerate(samples):
            sample["id"] = f"sample_{index}"
        dataset = samples[0]["dataset"]
        completed = {s["id"]: {"dataset": dataset, **trace_value(s)} for s in samples[::2]}
        store.merge_records(generation.collection_path("cot", dataset), dataset, completed)
        with patch.object(generation.baseline, "load_manifests", return_value=samples), \
             patch.object(generation.baseline, "create_engine", side_effect=AssertionError("must not load engine")):
            # Shard 0 contains only completed items, even after resume.
            self.assertEqual(generation.main(["--datasets", dataset, "--total-shards", "2", "--shard-id", "0"]), 0)
            # cot-only never silently runs Stage 1 for shard 1's missing evidence.
            self.assertEqual(generation.main(["--stage", "cot", "--datasets", dataset,
                                              "--total-shards", "2", "--shard-id", "1"]), 1)

    def test_stage2_all_formats_use_images_and_existing_validator(self):
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                sample = self.prepared_sample(modality, anomaly)
                evidence = generation.load_sample_evidence(sample, {sample["id"]: evidence_record(sample)})
                with patch.object(generation.baseline, "open_rgb", return_value=object()):
                    request, _ = generation.build_request(
                        sample, ProcessorStub(), config.STAGE2_PROMPT_PATH.read_text(), "cot", evidence,
                    )
                self.assertTrue(request["multi_modal_data"]["image"])
                branch = "NORMAL" if not anomaly else ("ANOMALOUS_2D" if modality == "2d" else "ANOMALOUS_MULTIVIEW")
                self.assertIn("Required response branch: " + branch, request["prompt"])
                self.assertTrue(generation.validate_reasoning(trace_value(sample)["reasoning"], sample, min_iou=0.9).valid)

    def test_default_full_run_single_engine_disk_handoff_and_resume(self):
        samples = [self.prepared_sample(anomaly=False) for _ in config.DATASETS]
        for sample, dataset in zip(samples, config.DATASETS):
            sample["dataset"] = dataset
            sample["id"] = dataset + "_sample"
        responses = [json.dumps(evidence_value(visibility="not_visible")) for _ in samples]
        responses += [json.dumps(trace_value(sample)) for sample in samples]
        engine = EngineStub(responses)
        original_generate = engine.generate

        def generate_after_disk_check(requests, stage):
            if stage == "cot":
                # All Stage 1 records must exist on disk BEFORE Stage 2 starts.
                for sample in samples:
                    persisted = store.read_records(generation.collection_path("evidence", sample["dataset"]), sample["dataset"])
                    self.assertTrue(generation.has_evidence(sample, persisted))
            return original_generate(requests, stage)

        engine.generate = generate_after_disk_check
        processor = ProcessorStub()
        transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)))
        with patch.dict("sys.modules", {"transformers": transformers}), \
             patch.object(generation.baseline, "load_manifests", return_value=samples), \
             patch.object(generation.baseline, "open_rgb", return_value=object()), \
             patch.object(generation.baseline, "create_engine", return_value=engine) as create_engine, \
             patch.object(generation, "make_sampling_params", side_effect=lambda args, stage: stage) as sampling:
            self.assertEqual(generation.main([]), 0)
            self.assertEqual(create_engine.call_count, 1)
            self.assertEqual(sampling.call_count, 2)
            self.assertEqual(len(engine.requests), 8)
            self.assertEqual(generation.main([]), 0)
            self.assertEqual(create_engine.call_count, 1)  # Fully complete rerun loads nothing.
        for sample in samples:
            records = store.read_records(generation.collection_path("cot", sample["dataset"]), sample["dataset"])
            self.assertTrue(generation.has_trace(sample, records, 0.9))

    def test_sampling_schemas_are_independent(self):
        sampling = Mock(side_effect=lambda **kwargs: kwargs)
        structured = Mock(side_effect=lambda **kwargs: kwargs)
        modules = {
            "vllm": SimpleNamespace(SamplingParams=sampling),
            "vllm.sampling_params": SimpleNamespace(StructuredOutputsParams=structured),
        }
        with patch.dict("sys.modules", modules):
            args = generation.parse_args([])
            evidence = generation.make_sampling_params(args, "evidence")
            cot = generation.make_sampling_params(args, "cot")
        self.assertEqual(evidence["structured_outputs"]["json"]["title"], "VisualEvidence")
        self.assertEqual(cot["structured_outputs"]["json"]["title"], "ReasoningTrace")
        self.assertEqual((evidence["temperature"], evidence["max_tokens"]), (0.2, 2048))

    def test_compilation_reads_dataset_trace_and_exports_query_only(self):
        sample = self.prepared_sample()
        dataset = sample["dataset"]
        config.MANIFEST_DIR.mkdir()
        (config.MANIFEST_DIR / f"{dataset}.json").write_text(json.dumps({"samples": [sample]}))
        with self.assertRaises(RuntimeError):
            compile_dataset(dataset, allow_missing=False, min_iou=0.9, trace_source="evidence-first")
        store.merge_records(generation.collection_path("cot", dataset), dataset,
                            {sample["id"]: {"dataset": dataset, **trace_value(sample)}})
        self.assertEqual(compile_dataset(dataset, allow_missing=False, min_iou=0.9, trace_source="evidence-first"), (1, 0))
        output = json.loads((config.EVIDENCE_FIRST_COMPILED_DIR / f"{dataset}_cot_sft.json").read_text())
        self.assertEqual(output[0]["images"], [s["path"] for s in sample["query_images"]])
        self.assertEqual(output[0]["messages"][1]["content"], trace_value(sample)["reasoning"])
        self.assertFalse(config.COMPILED_DIR.exists())
        # The existing compiler defaults still read the original per-sample files.
        directory = config.TRACE_DIR / dataset
        directory.mkdir(parents=True)
        (directory / f"trace_{sample['id']}.json").write_text(json.dumps(trace_value(sample)))
        self.assertEqual(compile_dataset(dataset, allow_missing=False, min_iou=0.9), (1, 0))


if __name__ == "__main__":
    unittest.main()
