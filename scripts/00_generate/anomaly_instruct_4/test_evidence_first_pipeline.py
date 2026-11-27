"""No model weights, CUDA, or vLLM required: exercise the two-stage data flow."""

import copy
import json
import multiprocessing
import re
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
    boxes = sample["supervision"]["bboxes"]
    view_id = boxes[0]["view_id"] if boxes else sample["query_images"][0]["view_id"]
    return {"image_id": sample["id"], "dataset": sample["dataset"],
            "evidence": evidence_value(view_id, "not_visible")}


def model_value(sample, *, words=90, regions=1):
    anomaly = sample["is_anomaly"]
    return {"evidence_review": [
        {"region_index": index, "decision": "revise",
         "verified_observation": ("The upper rim bends inward at one spot." if anomaly else
                                  "The inspected rim has a continuous, even contour.")}
        for index in range(regions)
    ], "inspection_conclusion": "anomalous" if anomaly else "normal",
        "supporting_region_index": 0 if anomaly else None,
        "inspection_text": " ".join(f"visible{index}" for index in range(words))}


def trace_value(sample, **kwargs):
    value = model_value(sample, **kwargs)
    return {**value, "image_id": sample["id"],
            "reasoning": generation.assemble_reasoning(value["inspection_text"], sample)}


class ProcessorStub:
    def __init__(self):
        self.messages = []
        self.template_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        self.template_kwargs.append(kwargs)
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
                self.assertTrue(processor.template_kwargs[-1]["add_vision_id"])
                self.assertNotIn("centered near", prompt)
                if anomaly:
                    view_id = sample["supervision"]["bboxes"][0]["view_id"]
                    self.assertIn(f"Priority views with marked areas: {view_id}.", prompt)
                else:
                    self.assertNotIn("Priority views with marked areas", prompt)
                if modality == "3d_multiview" and anomaly:
                    self.assertIn("Picture 1: clean original query (view_id=view_000)", prompt)
                    self.assertIn("Picture 2: localization cue (view_id=view_000)", prompt)
                    self.assertIn("Picture 3: same-category normal reference (view_id=view_000)", prompt)

    def test_evidence_preserves_uncertainty_and_view_mapping(self):
        sample = self.prepared_sample("3d_multiview")
        value = evidence_value("view_000", "not_visible")
        with self.assertRaisesRegex(ValueError, generation.FOCUS_COVERAGE_ERROR):
            generation.parse_evidence(json.dumps(value), sample)
        marked = evidence_value("view_017")["regional_observations"][0]
        marked["query_evidence"] = []
        value["regional_observations"].append(marked)
        with self.assertRaisesRegex(ValueError, generation.FOCUS_COVERAGE_ERROR):
            generation.parse_evidence(json.dumps(value), sample)
        marked["visibility"] = "not_visible"
        limit = marked["uncertainty"]
        marked["uncertainty"] = "none"
        with self.assertRaisesRegex(ValueError, generation.FOCUS_COVERAGE_ERROR):
            generation.parse_evidence(json.dumps(value), sample)
        marked["uncertainty"] = limit
        self.assertEqual(generation.parse_evidence(json.dumps(value), sample).model_dump(), value)
        # view_000 has a cue image, but only view_017 has an actual GT box.
        text = generation.format_stage1_user_prompt(sample)
        self.assertIn("view_id=view_000", text)
        self.assertIn("Priority views with marked areas: view_017.", text)
        self.assertNotIn("view_017 is anomalous", text)
        value["regional_observations"][0]["view_id"] = "view_999"
        with self.assertRaises(ValueError):
            generation.parse_evidence(json.dumps(value), sample)

    def test_internal_sources_allowed_but_final_sources_rejected_in_all_branches(self):
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                sample = self.prepared_sample(modality, anomaly)
                value = evidence_record(sample)["evidence"]
                value["regional_observations"].append(copy.deepcopy(value["regional_observations"][0]))
                for mention in ("See Picture 3.", "Compared to the normal reference (view_id=query).", "view_017"):
                    for field in ("object_observation", "region", "query_evidence", "normal_comparison", "uncertainty"):
                        candidate = copy.deepcopy(value)
                        target = candidate if field == "object_observation" else candidate["regional_observations"][0]
                        target[field] = [mention] if field == "query_evidence" else mention
                        self.assertEqual(generation.parse_evidence(json.dumps(candidate), sample).model_dump(), candidate)
                    body = mention + " The visible boundary has an angular interruption."
                    result = generation.validate_evidence_first_reasoning(generation.assemble_reasoning(body, sample), sample)
                    self.assertFalse(result.valid)
                body = "Four holes and a 3 mm mark are visible along the edge."
                self.assertTrue(generation.validate_evidence_first_reasoning(generation.assemble_reasoning(body, sample), sample).valid)

    def test_internal_comparison_is_saved_and_final_picture_reference_is_retried(self):
        sample = self.prepared_sample()
        value = evidence_value()
        value["regional_observations"][0]["normal_comparison"] = "See Picture 3: the corresponding contour is more even."
        processor = ProcessorStub()
        args = generation.parse_args(["--max-retries", "2"])
        with patch.object(generation.baseline, "open_rgb", return_value=object()), \
             patch.object(generation, "make_sampling_params", return_value=SimpleNamespace(max_tokens=2048)):
            self.assertEqual(generation.process_chunk([sample], EngineStub([json.dumps(value)]), processor, None,
                                                      "system", args, "evidence", {}), (1, 0))
            records = store.read_records(generation.collection_path("evidence", sample["dataset"]), sample["dataset"])
            self.assertEqual(records[sample["id"]]["attempt"], 1)
            good = model_value(sample)
            good["evidence_review"][0]["verified_observation"] = "The original edge is irregular compared to the normal reference."
            bad = {**good, "inspection_text": "See Picture 3. " + good["inspection_text"]}
            repair = {"inspection_text": good["inspection_text"]}
            engine = EngineStub([json.dumps(bad), json.dumps(repair)])
            self.assertEqual(generation.process_chunk([sample], engine, processor, None, "system", args, "cot", records),
                             (1, 0))
            self.assertIn("Remove picture numbers", processor.messages[-1][1]["content"][-1]["text"])
            self.assertIn("multi_modal_data", engine.requests[0][0])
            self.assertNotIn("multi_modal_data", engine.requests[1][0])
            trace = store.read_records(generation.collection_path("cot", sample["dataset"]), sample["dataset"])[sample["id"]]
            self.assertEqual(trace["evidence_review"], good["evidence_review"])
            self.assertEqual(trace["inspection_conclusion"], good["inspection_conclusion"])
        prompt = generation.format_stage1_user_prompt(sample, ["private /broken/123.png [101,52,203,177]"])
        self.assertNotIn("/broken/123.png", prompt)
        self.assertNotIn("[101,52,203,177]", prompt)

    def test_stage1_view_mentions_do_not_override_structured_view_id(self):
        sample = self.prepared_sample("3d_multiview")
        value = evidence_value("view_017")
        value["regional_observations"][0]["normal_comparison"] += " (view_id=view_017)"
        self.assertEqual(generation.parse_evidence(json.dumps(value), sample).model_dump(), value)
        value["regional_observations"][0]["view_id"] = "view_999"
        with self.assertRaisesRegex(ValueError, "unknown query view_id"):
            generation.parse_evidence(json.dumps(value), sample)

    def test_stage1_retries_definitive_comparison_judgments(self):
        sample = self.prepared_sample()
        good = evidence_value()
        good["regional_observations"][0]["normal_comparison"] = (
            "The corresponding edge is continuous. This view does not confirm damage."
        )
        bad = copy.deepcopy(good)
        for verdict in ("The comparison confirms the presence of damage in the query.",
                        "This confirms that the irregular edge in the query is an anomaly."):
            bad["regional_observations"][0]["normal_comparison"] = verdict
            with self.assertRaisesRegex(ValueError, generation.COMPARISON_VERDICT_ERROR):
                generation.parse_evidence(json.dumps(bad), sample)
        processor = ProcessorStub()
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            result = generation.process_chunk(
                [sample], EngineStub([json.dumps(bad), json.dumps(good)]), processor, None,
                "system", generation.parse_args(["--max-retries", "2"]), "evidence", {},
            )
        self.assertEqual(result, (1, 0))
        self.assertIn("Rewrite normal_comparison", processor.messages[-1][1]["content"][-1]["text"])
        records = store.read_records(generation.collection_path("evidence", sample["dataset"]), sample["dataset"])
        self.assertEqual(records[sample["id"]]["evidence"], good)

    def test_stage1_retries_region_and_comparison_placeholders(self):
        sample = self.prepared_sample()
        good = evidence_value()
        observation = good["regional_observations"][0]
        observation["region"] = "Upper-left edge of the opening"
        observation["normal_comparison"] = "The corresponding edge is hidden by reflection in the reference."
        invalid = []
        for field, placeholder, error in (
            ("region", "centered near the middle row, center column of the full query", generation.REGION_LOCATOR_ERROR),
            ("region", "red box", generation.BBOX_CUE_ERROR),
            ("normal_comparison", "Not applicable; this is a defect observation in the query view.", generation.COMPARISON_CONTENT_ERROR),
            ("normal_comparison", "N/A", generation.COMPARISON_CONTENT_ERROR),
        ):
            bad = copy.deepcopy(good)
            bad["regional_observations"][0][field] = placeholder
            with self.subTest(field=field, placeholder=placeholder):
                with self.assertRaisesRegex(ValueError, error):
                    generation.parse_evidence(json.dumps(bad), sample)
            invalid.append(bad)
        processor = ProcessorStub()
        engine = EngineStub([json.dumps(item) for item in (invalid[0], invalid[2], good)])
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            result = generation.process_chunk(
                [sample], engine, processor, None, "system",
                generation.parse_args(["--max-retries", "3"]), "evidence", {},
            )
        self.assertEqual(result, (1, 0))
        self.assertIn("Name the inspected part or area in region", engine.requests[1][0]["prompt"])
        self.assertIn("Compare the same part in query and reference", engine.requests[2][0]["prompt"])
        records = store.read_records(generation.collection_path("evidence", sample["dataset"]), sample["dataset"])
        self.assertEqual(records[sample["id"]]["evidence"], good)

    def test_stage1_region_wording_is_not_a_rejection_rule(self):
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                sample = self.prepared_sample(modality, anomaly)
                value = evidence_record(sample)["evidence"]
                for region in ("Upper rim", "The upper rim shows an angular interruption.",
                               "Outer edge exhibiting uneven texture.", "lower-left quadrant", "central area"):
                    value["regional_observations"][0]["region"] = region
                    self.assertEqual(generation.parse_evidence(json.dumps(value), sample).model_dump(), value)

    def test_stage1_removes_only_a_trailing_cue_from_a_located_region(self):
        sample = self.prepared_sample()
        clean = evidence_value()
        clean["regional_observations"][0]["region"] = "Upper-left edge of the opening"
        raw = copy.deepcopy(clean)
        tail = ", as indicated by the red box in Picture 2."
        raw["regional_observations"][0]["region"] += tail
        self.assertEqual(generation.parse_evidence(json.dumps(raw), sample).model_dump(), clean)
        raw["regional_observations"][0]["region"] = "The highlighted area" + tail
        with self.assertRaisesRegex(ValueError, generation.BBOX_CUE_ERROR):
            generation.parse_evidence(json.dumps(raw), sample)

    def test_stage1_retry_names_the_fields_with_cue_leakage(self):
        sample = self.prepared_sample()
        good = evidence_value()
        bad = copy.deepcopy(good)
        observation = bad["regional_observations"][0]
        observation["query_evidence"] = ["The edge follows the red box."]
        observation["normal_comparison"] = "The reference lacks the red outline cue."
        engine = EngineStub([json.dumps(bad), json.dumps(good)])
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            result = generation.process_chunk(
                [sample], engine, ProcessorStub(), None, "system",
                generation.parse_args(["--max-retries", "2"]), "evidence", {},
            )
        self.assertEqual(result, (1, 0))
        prompt = engine.requests[1][0]["prompt"]
        self.assertIn("Rewrite query_evidence using the clean query", prompt)
        self.assertIn("Redo normal_comparison using the clean query and the listed same-category normal reference", prompt)
        self.assertNotIn("The reference lacks the red outline cue.", prompt)
        records = store.read_records(generation.collection_path("evidence", sample["dataset"]), sample["dataset"])
        self.assertEqual(records[sample["id"]]["generation_attempts"][0]["errors"], [
            generation.BBOX_CUE_ERROR,
            f"{generation.BBOX_CUE_ERROR} in query_evidence",
            f"{generation.BBOX_CUE_ERROR} in normal_comparison",
        ])

    def test_annotation_cues_are_rejected_in_descriptive_fields(self):
        sample = self.prepared_sample()
        value = evidence_value()
        value["regional_observations"][0]["query_evidence"] = ["A red printed stripe runs along the rim."]
        evidence = generation.parse_evidence(json.dumps(value), sample)
        leaked = "The reference lacks the red outline cue."
        for field in ("object_observation", "region", "query_evidence", "normal_comparison", "uncertainty"):
            bad = copy.deepcopy(value)
            target = bad if field == "object_observation" else bad["regional_observations"][0]
            target[field] = [leaked] if field == "query_evidence" else leaked
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, generation.BBOX_CUE_ERROR):
                generation.parse_evidence(json.dumps(bad), sample)
        trace = trace_value(sample)
        trace["evidence_review"][0]["verified_observation"] = leaked
        model_fields = generation.EvidenceAnchoredTrace.model_fields
        raw = json.dumps({field: trace[field] for field in model_fields})
        with self.assertRaisesRegex(ValueError, generation.BBOX_CUE_ERROR):
            generation.parse_stage2_response(raw, evidence)
        records = {sample["id"]: {**trace, "dataset": sample["dataset"]}}
        self.assertFalse(generation.has_trace(sample, records, 0.9))
        result = generation.validate_evidence_first_reasoning(generation.assemble_reasoning(leaked, sample), sample)
        self.assertIn(generation.BBOX_CUE_ERROR, result.errors)

    def test_stage1_preserves_useful_evidence_without_wording_retry(self):
        sample = self.prepared_sample()
        value = evidence_value()
        value["regional_observations"][0]["region"] = "The upper rim shows an angular interruption."
        processor = ProcessorStub()
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            self.assertEqual(generation.process_chunk([sample], EngineStub([json.dumps(value)]), processor, None,
                             "system", generation.parse_args([]), "evidence", {}), (1, 0))
        saved = store.read_records(generation.collection_path("evidence", sample["dataset"]), sample["dataset"])
        self.assertEqual(saved[sample["id"]]["evidence"], value)
        self.assertEqual(saved[sample["id"]]["attempt"], 1)

    def test_collection_concurrent_appends(self):
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
        self.assertEqual(len(json.loads(path.read_bytes())["records"]), 15)
        with self.assertRaises(ValueError):
            store.read_records(path, "wrong_dataset")

    def test_legacy_collection_conversion_is_atomic_and_preserves_records(self):
        path = self.root / "evidence_test.json"
        old = {"old": {"image_id": "old", "dataset": "test"}}
        path.write_text(json.dumps({"schema_version": "anomaly-instruct-dataset-records-v1", "dataset": "test", "records": old}))
        self.assertEqual(store.read_records(path, "test"), old)
        original = path.read_bytes()
        with patch.object(store.os, "replace", side_effect=OSError("simulated interrupted write")):
            with self.assertRaises(OSError):
                store.merge_records(path, "test", {"new": {"image_id": "new", "dataset": "test"}})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        store.merge_records(path, "test", {"new": {"image_id": "new", "dataset": "test"}})
        self.assertEqual(set(store.read_records(path, "test")), {"old", "new"})
        self.assertIsInstance(json.loads(path.read_bytes())["records"], list)

    def test_append_only_writes_new_bytes_and_latest_record_wins(self):
        path = self.root / "evidence_test.json"
        old = {"image_id": "same", "dataset": "test", "value": "旧记录\n含有 ]} 和引号\""}
        store.merge_records(path, "test", {"same": old})
        before, inode = path.read_bytes(), path.stat().st_ino
        updated = {**old, "value": "new version"}
        new = {"image_id": "new", "dataset": "test"}
        # Appending must neither parse all old records nor replace the file.
        with patch.object(store, "_read_unlocked", side_effect=AssertionError("full read")), \
             patch.object(store, "_replace_snapshot", side_effect=AssertionError("full rewrite")):
            store.merge_records(path, "test", {"same": updated, "new": new})
        self.assertEqual(path.stat().st_ino, inode)
        self.assertTrue(path.read_bytes().startswith(before[:-len(store.FOOTER)]))
        self.assertEqual(store.read_records(path, "test"), {"same": updated, "new": new})
        self.assertEqual(len(json.loads(path.read_bytes())["records"]), 3)

    def test_recover_every_interrupted_append_position_for_readers_and_writers(self):
        path = self.root / "evidence_test.json"
        saved = {"image_id": "saved", "dataset": "test"}
        fresh = {"image_id": "fresh", "dataset": "test", "value": "中文\n含引号\"和 ]}"}
        later = {"image_id": "later", "dataset": "test"}
        store.merge_records(path, "test", {"saved": saved})
        prefix = path.read_bytes()[:-len(store.FOOTER)]
        append = b"," + store._encode_records({"fresh": fresh}, "test") + store.FOOTER
        with patch.object(store.logging.getLogger(store.__name__), "warning"):
            for reader_first in (True, False):
                for cut in range(len(append) + 1):
                    with self.subTest(reader_first=reader_first, cut=cut):
                        path.write_bytes(prefix + append[:cut])
                        if reader_first:
                            self.assertEqual(store.read_records(path, "test")["saved"], saved)
                            json.loads(path.read_bytes())  # Recovery leaves standard JSON.
                        store.merge_records(path, "test", {"later": later})
                        recovered = store.read_records(path, "test")
                        self.assertEqual(recovered["saved"], saved)
                        self.assertEqual(recovered["later"], later)
                        if "fresh" in recovered:
                            self.assertEqual(recovered["fresh"], fresh)

    def test_recovery_rejects_complete_corruption_without_modifying_file(self):
        path = self.root / "evidence_test.json"
        store.merge_records(path, "test", {"saved": {"image_id": "saved", "dataset": "test"}})
        prefix = path.read_bytes()[:-len(store.FOOTER)]
        invalid = [b',{"broken":!}\n', b',{"image_id":"other","dataset":"wrong"}\n']
        for tail in invalid:
            path.write_bytes(prefix + tail)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                store.read_records(path, "test")
            with self.assertRaises(ValueError):
                store.merge_records(path, "test", {"new": {"image_id": "new", "dataset": "test"}})
            self.assertEqual(path.read_bytes(), original)

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
                [sample], EngineStub([json.dumps(model_value(sample))]), processor, None, "system", args, "cot",
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
            self.assertFalse((config.EVIDENCE_FIRST_TRACE_DIR / "failed" / "cot_shard_1.jsonl").exists())
            log = (config.LOG_DIR / "evidence_first_shard_1.log").read_text()
            self.assertIn("generation failures=0; CoT samples waiting for Stage 1 evidence=2", log)

    def test_stage2_all_formats_use_images_and_existing_validator(self):
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                sample = self.prepared_sample(modality, anomaly)
                evidence = generation.load_sample_evidence(sample, {sample["id"]: evidence_record(sample)})
                processor = ProcessorStub()
                with patch.object(generation.baseline, "open_rgb", return_value=object()):
                    request, _ = generation.build_request(
                        sample, processor, config.STAGE2_PROMPT_PATH.read_text(), "cot", evidence,
                    )
                self.assertTrue(request["multi_modal_data"]["image"])
                self.assertTrue(processor.template_kwargs[-1]["add_vision_id"])
                for private in (sample["id"], "Ground Truth:", "Required response branch:",
                                "Canonical anomaly type:", "[101,52,203,177]"):
                    self.assertNotIn(private, request["prompt"])
                if modality == "3d_multiview" and anomaly:
                    self.assertIn("Picture 1: clean original query (view_id=view_000)", request["prompt"])
                    self.assertIn("Picture 2: localization cue (view_id=view_000)", request["prompt"])
                    self.assertIn("Picture 3: same-category normal reference (view_id=view_000)", request["prompt"])
                    self.assertIn("Priority views with marked areas: view_017.", request["prompt"])
                    self.assertNotIn("centered near", request["prompt"])
                self.assertTrue(generation.validate_evidence_first_reasoning(trace_value(sample)["reasoning"], sample, min_iou=0.9).valid)

    def test_default_full_run_single_engine_disk_handoff_and_resume(self):
        samples = [self.prepared_sample(anomaly=False) for _ in config.DATASETS]
        for sample, dataset in zip(samples, config.DATASETS):
            sample["dataset"] = dataset
            sample["id"] = dataset + "_sample"
        responses = [json.dumps(evidence_value(visibility="not_visible")) for _ in samples]
        responses += [json.dumps(model_value(sample)) for sample in samples]
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

    def test_sampling_schemas_are_independent_and_use_plain_json_strings(self):
        modules = {"vllm": SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
                   "vllm.sampling_params": SimpleNamespace(StructuredOutputsParams=lambda **kwargs: kwargs)}
        with patch.dict("sys.modules", modules), patch.dict("os.environ", {}, clear=True):
            args = generation.parse_args([])
            evidence = generation.make_sampling_params(args, "evidence")
            cot = generation.make_sampling_params(args, "cot")
        self.assertEqual(evidence["structured_outputs"]["json"]["title"], "VisualEvidence")
        self.assertNotIn("minLength", json.dumps(evidence["structured_outputs"]["json"]))
        invalid_evidence = evidence_value()
        invalid_evidence["object_observation"] = ""
        with self.assertRaises(ValueError):
            generation.parse_evidence(json.dumps(invalid_evidence), self.prepared_sample())
        schema = cot["structured_outputs"]["json"]
        self.assertEqual(schema["required"], ["evidence_review", "inspection_conclusion", "supporting_region_index", "inspection_text"])
        self.assertNotIn("pattern", schema["properties"]["inspection_text"])
        self.assertIn("130", schema["properties"]["inspection_text"]["description"])
        self.assertNotIn("reasoning", schema["properties"])
        self.assertEqual((cot["temperature"], cot["presence_penalty"], cot["repetition_penalty"]), (0.2, 0, 1.05))
        self.assertEqual(evidence["max_tokens"], 2048)

    def test_program_formats_all_branches_from_supervision_without_model_tags(self):
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                sample = self.prepared_sample(modality, anomaly)
                evidence = generation.load_sample_evidence(sample, {sample["id"]: evidence_record(sample)})
                prompt = generation.format_stage2_user_prompt(sample, evidence)
                self.assertIn("regional_observations", prompt)
                self.assertIn('"region_index": 0', prompt)
                self.assertIn(f'"priority_view": {str(anomaly).lower()}', prompt)
                self.assertNotIn(evidence.object_observation, prompt)
                self.assertNotIn("short plain-text", prompt)
                self.assertNotIn("WRITE_", prompt)
                self.assertNotIn('"decision": "keep"', prompt)
                value = model_value(sample)
                trace = generation.parse_stage2_response(json.dumps(value), evidence)
                reasoning = generation.assemble_reasoning(trace.inspection_text, sample)
                self.assertTrue(generation.validate_evidence_first_reasoning(reasoning, sample, min_iou=0.9).valid)
                if anomaly:
                    locations = re.search(r"<location>(.*?)</location>", reasoning).group(1)
                    self.assertEqual(json.loads(locations), generation.baseline.ground_truth_location_value(sample))
                    self.assertIn("<type>concavity</type>", reasoning)
                else:
                    self.assertNotIn("<location>", reasoning)
                    self.assertNotIn("<type>", reasoning)

    def test_placeholder_is_retried_without_rewriting_evidence(self):
        sample = self.prepared_sample()
        path = generation.collection_path("evidence", sample["dataset"])
        records = {sample["id"]: evidence_record(sample)}
        store.merge_records(path, sample["dataset"], records)
        before = path.read_bytes()
        processor = ProcessorStub()
        bad = {**model_value(sample), "inspection_text": "visually grounded inspection reasoning"}
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            result = generation.process_chunk([sample], EngineStub([json.dumps(bad), json.dumps(model_value(sample))]),
                processor, None, "system", generation.parse_args(["--max-retries", "2"]), "cot", records)
        self.assertEqual(result, (1, 0))
        self.assertIn("Replace placeholder", processor.messages[-1][1]["content"][-1]["text"])
        self.assertEqual(path.read_bytes(), before)
        saved = store.read_records(generation.collection_path("cot", sample["dataset"]), sample["dataset"])
        self.assertEqual(saved[sample["id"]]["attempt"], 2)
        self.assertTrue(generation.has_trace(sample, saved, 0.9))
        self.assertEqual(len(saved[sample["id"]]["generation_attempts"]), 2)

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

    def test_stage2_review_covers_every_region_exactly_once(self):
        sample = self.prepared_sample()
        value = evidence_value()
        value["regional_observations"].append({**value["regional_observations"][0], "region": "lower rim"})
        evidence = generation.parse_evidence(json.dumps(value), sample)
        trace = model_value(sample, regions=2)
        trace["evidence_review"][0]["decision"] = "revise"
        parsed = generation.parse_stage2_response(json.dumps(trace), evidence)
        self.assertEqual(parsed.evidence_review[0].decision, "revise")
        for indices, details in (
            ([0], ("missing region_index 1",)),
            ([0, 0], ("missing region_index 1", "duplicate region_index 0")),
            ([0, 2], ("missing region_index 1", "unexpected region_index 2")),
        ):
            invalid = copy.deepcopy(trace)
            invalid["evidence_review"] = [
                {**trace["evidence_review"][0], "region_index": index} for index in indices
            ]
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "evidence_review") as error:
                    generation.parse_stage2_response(json.dumps(invalid), evidence)
                prompt = generation.format_stage2_user_prompt(sample, evidence, [str(error.exception)])
                for detail in details:
                    self.assertIn(detail, prompt)

    def test_word_limits_and_resume_in_all_four_branches(self):
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                sample = self.prepared_sample(modality, anomaly)
                for words in (0, 1, 20, 79, 80, 130, 131):
                    with self.subTest(modality=modality, anomaly=anomaly, words=words):
                        trace = trace_value(sample, words=words)
                        result = generation.validate_evidence_first_reasoning(trace["reasoning"], sample)
                        self.assertEqual(result.word_count, words)
                        self.assertEqual(result.valid, 1 <= words <= 130)
                        records = {sample["id"]: {"dataset": sample["dataset"], **trace}}
                        self.assertEqual(generation.has_trace(sample, records, 0.9), 1 <= words <= 130)

    def test_overlong_think_retries_before_saving_and_leaves_evidence_unchanged(self):
        sample = self.prepared_sample()
        records = {sample["id"]: evidence_record(sample)}
        path = generation.collection_path("evidence", sample["dataset"])
        store.merge_records(path, sample["dataset"], records)
        before = path.read_bytes()
        processor = ProcessorStub()
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            result = generation.process_chunk(
                [sample], EngineStub([json.dumps(model_value(sample, words=131)),
                                      json.dumps(model_value(sample, words=130))]),
                processor, None, "system", generation.parse_args(["--max-retries", "2"]), "cot", records,
            )
        self.assertEqual(result, (1, 0))
        retry_prompt = processor.messages[-1][1]["content"][-1]["text"]
        self.assertIn("word count 131", retry_prompt)
        self.assertIn("1-130", retry_prompt)
        saved = store.read_records(generation.collection_path("cot", sample["dataset"]), sample["dataset"])
        self.assertEqual(saved[sample["id"]]["validation"]["word_count"], 130)
        self.assertEqual(saved[sample["id"]]["attempt"], 2)
        self.assertEqual(path.read_bytes(), before)

    def test_json_strings_keep_valid_quotes_and_newlines_and_reject_raw_controls(self):
        sample = self.prepared_sample()
        evidence = generation.parse_evidence(json.dumps(evidence_value()), sample)
        value = model_value(sample)
        value["inspection_text"] = 'The print reads "500".\nA narrow pale line crosses the surface.'
        value["evidence_review"][0]["verified_observation"] = 'Compared to the normal reference, the print "500" remains legible.'
        encoded = json.dumps(value)
        parsed = generation.parse_stage2_response(encoded, evidence)
        self.assertEqual(parsed.inspection_text, value["inspection_text"])
        self.assertTrue(generation.validate_evidence_first_reasoning(
            generation.assemble_reasoning(parsed.inspection_text, sample), sample).valid)
        for broken in (encoded.replace('\\n', '\n'), encoded.replace('\\"500\\"', '"500"')):
            with self.assertRaises(ValueError):
                generation.parse_stage2_response(broken, evidence)

    def test_observed_source_leaks_are_rejected_in_final_text_in_all_branches(self):
        phrases = {
            "matching the provided coordinates.": "coordinate leakage",
            "consistent across both provided views.": "reference leakage",
            "compared to the normal reference.": "reference leakage",
            "matching the canonical anomaly type.": "annotation leakage",
            "as recorded in evidence_review.": "pipeline leakage",
        }
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                sample = self.prepared_sample(modality, anomaly)
                for phrase, error in phrases.items():
                    with self.subTest(modality=modality, anomaly=anomaly, phrase=phrase):
                        body = "A narrow pale line is visible along the outer surface, " + phrase
                        result = generation.validate_evidence_first_reasoning(generation.assemble_reasoning(body, sample), sample)
                        self.assertIn(error, result.errors)

    def test_repeated_final_text_is_retried_but_matching_regions_are_allowed(self):
        sample = self.prepared_sample()
        value = evidence_value()
        value["regional_observations"].append(copy.deepcopy(value["regional_observations"][0]))
        records = {sample["id"]: {"image_id": sample["id"], "dataset": sample["dataset"], "evidence": value}}
        good = model_value(sample, words=20, regions=2)
        phrase = "The outer edge has a narrow pale line and an uneven boundary. "
        bad = {**good, "inspection_text": phrase * 6}
        self.assertTrue(generation.has_repetition(bad["inspection_text"]))
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            self.assertEqual(generation.process_chunk([sample], EngineStub([json.dumps(bad), json.dumps(good)]),
                             ProcessorStub(), None, "system", generation.parse_args([]), "cot", records), (1, 0))
        saved = store.read_records(generation.collection_path("cot", sample["dataset"]), sample["dataset"])[sample["id"]]
        self.assertEqual(saved["attempt"], 2)
        self.assertIn("repetitive output in inspection_text", saved["generation_attempts"][0]["errors"])
        self.assertEqual(saved["evidence_review"][0]["verified_observation"], saved["evidence_review"][1]["verified_observation"])

    def test_truncation_retry_preserves_configured_budget_and_distinct_evidence(self):
        sample = self.prepared_sample()
        records = {sample["id"]: evidence_record(sample)}
        params = SimpleNamespace(max_tokens=2048)
        first = SimpleNamespace(outputs=[SimpleNamespace(text='{"unfinished":', finish_reason="length", stop_reason=None, token_ids=[1, 2])])
        second = SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(model_value(sample, words=20)), finish_reason="stop", stop_reason=151645, token_ids=[1, 2, 3])])
        engine = SimpleNamespace(generate=Mock(side_effect=[[first], [second]]))
        processor = ProcessorStub()
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            self.assertEqual(generation.process_chunk([sample], engine, processor, params, "system",
                             generation.parse_args([]), "cot", records), (1, 0))
        saved = store.read_records(generation.collection_path("cot", sample["dataset"]), sample["dataset"])[sample["id"]]
        self.assertEqual(saved["generation_attempts"][0]["finish_reason"], "length")
        self.assertEqual(saved["generation_attempts"][1]["output_tokens"], 3)
        self.assertEqual(saved["generation_attempts"][1]["token_budget"], 2048)
        self.assertEqual(params.max_tokens, 2048)
        self.assertEqual(saved["supervision_fields_source"], "manifest_gt")
        self.assertIn("preserving distinct visual facts", processor.messages[-1][1]["content"][-1]["text"])

    def test_short_and_long_valid_prose_can_be_saved(self):
        sample = self.prepared_sample()
        records = {sample["id"]: evidence_record(sample)}
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            for words in (40, 100, 125):
                with self.subTest(words=words):
                    result = generation.process_chunk([sample], EngineStub([json.dumps(model_value(sample, words=words))]),
                        ProcessorStub(), None, "system", generation.parse_args([]), "cot", records)
                    self.assertEqual(result, (1, 0))
                    saved = store.read_records(generation.collection_path("cot", sample["dataset"]), sample["dataset"])
                    record = saved[sample["id"]]
                    self.assertTrue(record["validation"]["valid"])
                    self.assertEqual(record["inspection_conclusion"], "anomalous")
                    self.assertEqual(record["supporting_region_index"], 0)
                    self.assertTrue(generation.has_trace(sample, saved, 0.9))

    def test_stage2_holds_anomalous_conclusion_without_verified_annotated_view(self):
        sample = self.prepared_sample("3d_multiview")
        record = evidence_record(sample)
        regions = record["evidence"]["regional_observations"]
        regions.append(evidence_value("view_000")["regional_observations"][0])
        for index, decision, error in (
            (None, "revise", "requires a valid supporting_region_index"),
            (0, "uncertain", "lacks a verified query observation"),
            (0, "keep", "lacks a verified query observation"),
            (1, "revise", "outside the annotated query views"),
        ):
            with self.subTest(index=index, decision=decision):
                value = model_value(sample, regions=2)
                value["supporting_region_index"] = index
                value["evidence_review"][0]["decision"] = decision
                engine = EngineStub([json.dumps(value)])
                with patch.object(generation.baseline, "open_rgb", return_value=object()):
                    result = generation.process_chunk(
                        [sample], engine, ProcessorStub(), None, "system",
                        generation.parse_args(["--max-retries", "2"]), "cot", {sample["id"]: record},
                    )
                self.assertEqual(result, (0, 1))
                self.assertEqual(len(engine.requests), 1)
                failures = config.EVIDENCE_FIRST_TRACE_DIR / "failed" / "cot_shard_0.jsonl"
                self.assertIn(error, failures.read_text().splitlines()[-1])
                self.assertFalse(generation.collection_path("cot", sample["dataset"]).exists())

    def test_stage2_holds_normal_conclusion_when_no_region_is_verified(self):
        sample = self.prepared_sample(anomaly=False)
        evidence = evidence_record(sample)
        value = model_value(sample)
        value["evidence_review"][0].update(decision="uncertain", verified_observation="Reflection hides the edge.")
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            result = generation.process_chunk(
                [sample], EngineStub([json.dumps(value)]), ProcessorStub(), None,
                "system", generation.parse_args([]), "cot", {sample["id"]: evidence},
            )
        self.assertEqual(result, (0, 1))
        failures = config.EVIDENCE_FIRST_TRACE_DIR / "failed" / "cot_shard_0.jsonl"
        self.assertIn("normal conclusion requires a verified observation", failures.read_text())

    def test_stage2_holds_label_disagreement_and_uncertainty_without_retry(self):
        for modality in ("2d", "3d_multiview"):
            for anomaly in (False, True):
                with self.subTest(modality=modality, anomaly=anomaly):
                    sample = self.prepared_sample(modality, anomaly)
                    for conclusion in ("normal" if anomaly else "anomalous", "uncertain"):
                        value = {**model_value(sample), "inspection_conclusion": conclusion}
                        engine = EngineStub([json.dumps(value)])
                        records = {sample["id"]: evidence_record(sample)}
                        with patch.object(generation.baseline, "open_rgb", return_value=object()):
                            result = generation.process_chunk([sample], engine, ProcessorStub(), None,
                                config.STAGE2_PROMPT_PATH.read_text(), generation.parse_args(["--max-retries", "2"]),
                                "cot", records)
                        self.assertEqual(result, (0, 1))
                        self.assertEqual(len(engine.requests), 1)
                        failures = (config.EVIDENCE_FIRST_TRACE_DIR / "failed" / "cot_shard_0.jsonl")
                        self.assertIn(f"inspection_conclusion: {conclusion}", failures.read_text())
                        self.assertFalse(generation.collection_path("cot", sample["dataset"]).exists())

    def test_stage2_retries_prose_contradiction_but_allows_local_negative(self):
        sample = self.prepared_sample()
        good = {**model_value(sample),
                "inspection_text": "The edge has a localized indentation. No cracks are visible."}
        bad = {**good, "inspection_text": "The overall condition suggests a standard, undamaged object."}
        engine = EngineStub([json.dumps(bad), json.dumps(good)])
        with patch.object(generation.baseline, "open_rgb", return_value=object()):
            result = generation.process_chunk([sample], engine, ProcessorStub(), None,
                config.STAGE2_PROMPT_PATH.read_text(), generation.parse_args(["--max-retries", "2"]),
                "cot", {sample["id"]: evidence_record(sample)})
        self.assertEqual(result, (1, 0))
        saved = store.read_records(generation.collection_path("cot", sample["dataset"]), sample["dataset"])
        self.assertEqual(saved[sample["id"]]["attempt"], 2)
        self.assertIn("No cracks are visible.", saved[sample["id"]]["reasoning"])
        self.assertTrue(generation.contradicts_inspection_conclusion("These findings indicate a defect.", "normal"))
        self.assertFalse(generation.contradicts_inspection_conclusion(
            "The outer rim has no visible defects.", "anomalous"))

    def test_compilation_rejects_overlong_evidence_first_think(self):
        sample = self.prepared_sample()
        dataset = sample["dataset"]
        config.MANIFEST_DIR.mkdir()
        (config.MANIFEST_DIR / f"{dataset}.json").write_text(json.dumps({"samples": [sample]}))
        trace = trace_value(sample, words=131)
        store.merge_records(generation.collection_path("cot", dataset), dataset,
                            {sample["id"]: {"dataset": dataset, **trace}})
        with self.assertRaisesRegex(RuntimeError, "missing/invalid"):
            compile_dataset(dataset, allow_missing=False, min_iou=0.9, trace_source="evidence-first")
        directory = config.TRACE_DIR / dataset
        directory.mkdir(parents=True)
        (directory / f"trace_{sample['id']}.json").write_text(json.dumps(trace))
        self.assertEqual(compile_dataset(dataset, allow_missing=False, min_iou=0.9), (1, 0))


if __name__ == "__main__":
    unittest.main()
