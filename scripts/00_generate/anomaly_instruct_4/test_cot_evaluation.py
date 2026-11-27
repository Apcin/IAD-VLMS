"""Offline integration checks: python -m unittest discover -s ... -p test_cot_evaluation.py."""
import contextlib
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

from PIL import Image

import evaluate_cot_glm as app


def response(score=5):
    result = {dimension: {"score": score, "reason": "充分支持", "checks": {
        key: {"verdict": "supported", "reason": "图像与文本一致"} for key in keys}}
        for dimension, keys in app.CHECKS.items()}
    result.update(issues=[], confidence="high", summary="与可见图像一致，推导连贯。")
    return {"id": "offline-test", "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(result), "reasoning_content": "offline reasoning"}}], "usage": {"total_tokens": 10}}


def concise_response(visual="忠实：绿色表面与完整轮廓有图像支持。", logic="合理：观察与正常结论一致，未推断不可见结构。",
                     *, visual_conflict=None):
    value = {"visual_faithfulness": visual, "reasoning_consistency": logic}
    if visual_conflict is not None:
        value["visual_conflict"] = visual_conflict
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value, ensure_ascii=False)}}]}


def color_conflict():
    return {"query_view_id": "query", "quote": "even green surface", "basis": "direct_query_conflict"}


def batch_response(count, entries=None):
    results = entries if entries is not None else [{"sample_index": i,
        "visual_faithfulness": f"忠实：本样本图像证据{i}清晰。", "reasoning_consistency": "合理：观察支持结论。"}
        for i in range(count)]
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"results": results}, ensure_ascii=False)}}],
            "usage": {"total_tokens": 200 * count}}


class EvaluationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.traces, self.manifests = self.root / "traces", self.root / "manifests"
        self.traces.mkdir(); self.manifests.mkdir()
        self.image = self.root / "query.png"
        Image.new("RGB", (64, 48), "green").save(self.image)
        self.ref = self.root / "normal.png"
        Image.new("RGB", (64, 48), "green").save(self.ref)
        for dataset in app.DATASETS:
            sample = {"id": dataset + "__sample", "dataset": dataset, "object_category": "object",
                      "is_anomaly": False, "question": "Are there defects?", "label": "Normal",
                      "query_images": [{"path": str(self.image), "view_id": "query", "width": 64, "height": 48}],
                      "normal_reference": {"images": [{"path": str(self.ref), "view_id": "query", "width": 64, "height": 48}]},
                      "supervision": {"bboxes": [], "bbox_prompt_images": [], "anomaly_type": None}}
            trace = {"image_id": sample["id"], "dataset": dataset,
                     "reasoning": "<think>The object has an even green surface.</think><answer>No</answer>",
                     "validation": {"valid": True}, "evidence_review": []}
            (self.manifests / f"{dataset}.json").write_text(json.dumps({"samples": [sample]}))
            # Latest append wins; a previous invalid record must not override a valid retry.
            (self.traces / f"traces_{dataset}.json").write_text(json.dumps({"records": [{**trace, "validation": {"valid": False}}, trace]}))
        self.args = app.parse_args(["evaluate", "--provider", "zhipu", "--trace-dir", str(self.traces), "--manifest-dir", str(self.manifests),
                  "--output-dir", str(self.root / "results"), "--workers", "1", "--request-interval", "0", "--retries", "1", "--port", "0"])
        app.STORE = app.SourceStore()
        app.encode_image.cache_clear()
        app.prepare(self.args)

    def test_catalog_deduplicates_paths_and_excludes_failed(self):
        with contextlib.closing(app.connect(self.args.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM samples WHERE active=1").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM assets").fetchone()[0], 2)
        failed = self.traces / "failed"; failed.mkdir()
        (failed / "cot_retry_2.jsonl").write_text(json.dumps({"dataset": "real3d", "image_id": "real3d__sample"})+"\n")
        app.prepare(self.args)
        with contextlib.closing(app.connect(self.args.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM samples WHERE active=1").fetchone()[0], 3)

    def test_request_uses_clean_query_reference_no_gt_paths_or_ids(self):
        with contextlib.closing(app.connect(self.args.db)) as db:
            ph, profile = app.get_profile(self.args, db)
            row, sample, trace = app.STORE.sample(db, 1)
        payload = app.make_request(sample, trace, profile)
        serialized = json.dumps(payload)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn(sample["id"], serialized)
        self.assertNotIn('"label"', serialized)
        images = [c for c in payload["messages"][1]["content"] if c["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertNotIn("response_format", payload)

    def a6_args(self, evaluation_format="scored"):
        return app.parse_args(["evaluate", "--trace-dir", str(self.traces), "--manifest-dir", str(self.manifests),
                              "--output-dir", str(self.args.output_dir), "--workers", "1", "--request-interval", "0",
                              "--evaluation-format", evaluation_format])

    def batch_members(self, db):
        return [app.STORE.sample(db, row[0]) for row in db.execute("SELECT id FROM samples ORDER BY dataset,image_id").fetchall()]

    def test_batch_request_has_boundaries_and_per_sample_output_budget(self):
        args = self.a6_args("concise")
        with contextlib.closing(app.connect(args.db)) as db:
            _, profile = app.get_profile(args, db)
            members = self.batch_members(db)
        payload = app.make_batch_request(members, profile)
        self.assertEqual(payload["max_tokens"], 4 * 512)
        self.assertNotIn("thinking", payload)
        content = payload["messages"][1]["content"]
        self.assertEqual(sum(c["type"] == "image_url" for c in content), 8)
        starts = [c["text"] for c in content if c["type"] == "text" and c["text"].startswith("SAMPLE_BEGIN")]
        self.assertEqual(starts, [f"SAMPLE_BEGIN sample_index={i}" for i in range(4)])
        serialized = json.dumps(payload)
        self.assertNotIn(str(self.root), serialized)
        for row, sample, trace in members:
            self.assertNotIn(row["image_id"], serialized)
        self.assertIn("最多100字", payload["messages"][0]["content"])
        with self.assertRaises(ValueError): app.make_batch_request([], profile)
        with self.assertRaises(ValueError): app.make_batch_request(members, {**profile, "max_tokens": 16384})

    def test_batch_packing_preserves_views_and_output_limits(self):
        ids = list(range(1,6))
        self.assertEqual(list(app.pack_sample_ids(ids, {i:8 for i in ids}, 4, 16, 512)), [[1,2],[3,4],[5]])
        self.assertEqual(list(app.pack_sample_ids(ids, {i:2 for i in ids}, 4, 32, 16384)), [[1,2],[3,4],[5]])
        with self.assertRaisesRegex(ValueError, "views will not be dropped"):
            list(app.pack_sample_ids(ids, {i:8 for i in ids}, 4, 4, 512))

    def test_batch_parser_maps_reordered_indices_and_keeps_partial_success(self):
        with contextlib.closing(app.connect(self.args.db)) as db:
            members = self.batch_members(db)
        entries = json.loads(batch_response(4)["choices"][0]["message"]["content"])["results"]
        valid, errors = app.parse_batch_evaluation(batch_response(4, list(reversed(entries))), members)
        self.assertFalse(errors)
        self.assertIn("证据0", valid[0]["visual_faithfulness"])
        valid, errors = app.parse_batch_evaluation(batch_response(4, [entries[0], entries[0], entries[1]]), members)
        self.assertEqual(set(valid), {1})
        self.assertEqual(set(errors), {0,2,3})
        entries[0]["sample_index"] = 99
        with self.assertRaisesRegex(ValueError, "sample_index"):
            app.parse_batch_evaluation(batch_response(4, entries), members)
        entries[0]["sample_index"] = False
        with self.assertRaises(ValueError):
            app.parse_batch_evaluation(batch_response(4, entries), members)

    def test_batch_evaluate_saves_raw_response_once_and_resumes_in_single_mode(self):
        args = self.a6_args("concise"); args.batch_size = 2
        with patch.object(app, "call_api", return_value=batch_response(2)) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 2)
        with contextlib.closing(app.connect(args.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM batch_attempts").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE status='done'").fetchone()[0], 4)
            attempts = db.execute("SELECT response FROM attempts").fetchall()
            self.assertEqual(len(attempts), 4)
            self.assertTrue(all(set(json.loads(r[0])) == {"batch_attempt_id"} for r in attempts))
            self.assertEqual(app.summary(db)["datasets"][0]["done"], 1)
        args.batch_size = 1
        with patch.object(app, "call_api") as mock:
            app._evaluate(args, "gateway-test-key")
            mock.assert_not_called()

    def test_partial_batch_retries_only_failed_members(self):
        args = self.a6_args("concise"); args.batch_size, args.retries = 4, 1
        entries = json.loads(batch_response(4)["choices"][0]["message"]["content"])["results"]
        entries[2]["visual_faithfulness"] = "bad format"
        first = batch_response(4, [entries[0], entries[2], entries[3]])
        with patch.object(app, "call_api", side_effect=[first, batch_response(2)]) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 2)
            second = mock.call_args_list[1].args[2]
            self.assertEqual(second["max_tokens"], 1024)
            self.assertEqual(sum(c["type"] == "image_url" for c in second["messages"][1]["content"]), 4)
        with contextlib.closing(app.connect(args.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE status='done'").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 6)
            infos = [json.loads(r[0])["members"] for r in db.execute("SELECT request_info FROM batch_attempts ORDER BY id")]
            self.assertEqual([m["sample_id"] for m in infos[1]], [infos[0][1]["sample_id"], infos[0][2]["sample_id"]])

    def test_exhausted_batch_keeps_valid_results_and_retries_missing_on_resume(self):
        args = self.a6_args("concise"); args.batch_size, args.retries = 4, 0
        with patch.object(app, "call_api", return_value=batch_response(3)):
            with self.assertRaisesRegex(ValueError, "1 samples failed"):
                app._evaluate(args, "gateway-test-key")
        with contextlib.closing(app.connect(args.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE status='done'").fetchone()[0], 3)
        with patch.object(app, "call_api", return_value=batch_response(1)) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(mock.call_args.args[2]["max_tokens"], 512)

    def test_batch_workers_issue_concurrent_requests(self):
        args = self.a6_args("concise"); args.batch_size, args.workers = 2, 2
        barrier = threading.Barrier(2)
        threads = set()
        lock = threading.Lock()
        def request(*_):
            with lock: threads.add(threading.get_ident())
            barrier.wait(timeout=5)
            return batch_response(2)
        with patch.object(app, "call_api", side_effect=request) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 2)
            self.assertEqual(len(threads), 2)

    def test_batch_overload_uses_shared_cooldown(self):
        args = self.a6_args("concise"); args.batch_size, args.retries = 2, 1
        limiter = Mock(); limiter.wait.return_value = True; limiter.throttle.return_value = 30
        error = app.APIError("HTTP 429", retryable=True, rate_limited=True)
        with patch.object(app, "RateLimiter", return_value=limiter), patch.object(app, "call_api", side_effect=[error, batch_response(2), batch_response(2)]) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 3)
            limiter.throttle.assert_called_once_with(0)
            self.assertEqual(limiter.wait.call_count, 3)

    def test_batch_auth_failure_stops_future_groups(self):
        args = self.a6_args("concise"); args.batch_size, args.workers = 2, 1
        with patch.object(app, "call_api", side_effect=app.APIError("HTTP 401", fatal=True)) as mock:
            with self.assertRaises(ValueError):
                app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 1)
        with contextlib.closing(app.connect(args.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM batch_attempts").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE status='error'").fetchone()[0], 2)

    def test_bad_input_in_batch_does_not_fail_other_members(self):
        args = self.a6_args("concise"); args.batch_size, args.retries = 4, 0
        bad = self.root / "invalid.png"; bad.write_bytes(b"not an image")
        path = self.manifests / "mvtec3d.json"
        value = json.loads(path.read_text()); value["samples"][0]["query_images"][0]["path"] = str(bad)
        path.write_text(json.dumps(value)); app.prepare(args)
        with patch.object(app, "call_api", return_value=batch_response(3)) as mock:
            with self.assertRaisesRegex(ValueError, "1 samples failed"):
                app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 1)
        with contextlib.closing(app.connect(args.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE status='done'").fetchone()[0], 3)

    def test_batch_configuration_does_not_invalidate_completed_profile(self):
        args = self.a6_args("concise")
        with contextlib.closing(app.connect(args.db)) as db:
            before, _ = app.get_profile(args, db)
            args.batch_size, args.batch_max_images, args.workers = 4, 32, 2
            after, _ = app.get_profile(args, db)
            self.assertEqual(before, after)
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (["--provider", "zhipu", "--batch-size", "2"],
                          ["--evaluation-format", "scored", "--batch-size", "2"], ["--batch-size", "0"]):
                with self.assertRaises(SystemExit): app.parse_args(["evaluate", *flags])

    def test_concise_defaults_and_token_budget(self):
        defaults = app.parse_args(["evaluate"])
        self.assertEqual(defaults.evaluation_format, "concise")
        self.assertEqual(defaults.max_tokens, 512)
        args = self.a6_args("concise")
        with contextlib.closing(app.connect(args.db)) as db:
            _, profile = app.get_profile(args, db)
            _, sample, trace = app.STORE.sample(db, 1)
        payload = app.make_request(sample, trace, profile)
        self.assertEqual(profile["max_output_chars"], 100)
        self.assertTrue(profile["prompt"].startswith(app.CONCISE_PROMPT))
        self.assertIn("annotation_cue", profile["prompt"])
        self.assertEqual(payload["max_tokens"], 512)
        self.assertNotIn("thinking", payload)
        self.assertEqual(app.parse_args(["evaluate", "--evaluation-format", "scored"]).max_tokens, 16384)

    def test_concise_judgments_are_independent_and_schema_is_small(self):
        with contextlib.closing(app.connect(self.args.db)) as db:
            _, sample, trace = app.STORE.sample(db, 1)
        for visual, logic, expected in (("忠实：有图像支持。", "合理：推导一致。", "pass"),
                ("不忠实：颜色描述与图像矛盾。", "合理：假设下推导连贯。", "review"),
                ("忠实：颜色正确。", "不合理：观察与结论冲突。", "fail"),
                ("不确定：图像细节不清。", "合理：内部无矛盾。", "review")):
            value = app.parse_evaluation(concise_response(visual, logic), trace, sample, "concise")
            self.assertEqual(value["verdict"], expected)
            self.assertNotIn("score", value)
            self.assertNotIn("checks", value)
            self.assertLessEqual(value["text_chars"], 100)
        for bad in (response(), concise_response("看不清", "合理：内部一致。"), concise_response("忠实：", "合理：内部一致。")):
            with self.assertRaises(ValueError):
                app.parse_evaluation(bad, trace, sample, "concise")

    def test_long_concise_output_is_bounded_without_extra_paid_requests(self):
        args = self.a6_args("concise")
        raw = concise_response("忠实：" + "观察清晰。" * 35, "合理：" + "推导一致。" * 30)
        with patch.object(app, "call_api", return_value=raw) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 4)
        with contextlib.closing(app.connect(args.db)) as db:
            row = db.execute("SELECT * FROM results WHERE profile=? LIMIT 1", (app.active_profile(db),)).fetchone()
            value = json.loads(row["evaluation"])
            self.assertIsNone(row["visual_score"])
            self.assertIsNone(row["logic_score"])
            self.assertTrue(value["length_limited"])
            self.assertEqual(sum(len(value[d]) for d in app.CONCISE_JUDGMENTS), 100)
            self.assertEqual(json.loads(db.execute("SELECT response FROM attempts LIMIT 1").fetchone()[0]), raw)

    def test_visual_rejection_requires_supported_citation_in_batch(self):
        args = self.a6_args("concise")
        args.batch_size = 4
        entries = [{"sample_index": i, "visual_faithfulness": "不忠实：查询图表面颜色与描述不符。",
                    "reasoning_consistency": "合理：观察到结论自洽。", "visual_conflict": conflict}
                   for i, conflict in enumerate([
                       color_conflict(),
                       {**color_conflict(), "query_view_id": "view_999"},
                       {**color_conflict(), "quote": "invented words"},
                       {**color_conflict(), "basis": "reference_only"},
                   ])]
        # An invalid visual citation must not override an independent reasoning error.
        entries[3]["reasoning_consistency"] = "不合理：正文与答案冲突。"
        with patch.object(app, "call_api", return_value=batch_response(4, entries)) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 1)
        with contextlib.closing(app.connect(args.db)) as db:
            results = [json.loads(row[0]) for row in db.execute("""
              SELECT r.evaluation FROM results r JOIN samples s ON s.id=r.sample_id ORDER BY s.dataset,s.image_id
            """)]
        self.assertEqual([r["verdict"] for r in results], ["fail", "review", "review", "fail"])
        self.assertEqual(results[0]["visual_conflict"], color_conflict())
        self.assertTrue(all(r["visual_conflict_unverified"] for r in results[1:]))
        self.assertTrue(all(r["judgments"]["visual_faithfulness"] == "不确定" for r in results[1:]))

    def test_format_change_reuses_paid_results_and_force_requests_short_judgments(self):
        scored = self.a6_args("scored")
        with patch.object(app, "call_api", return_value=response()):
            app._evaluate(scored, "gateway-test-key")
        short = self.a6_args("concise")
        with patch.object(app, "call_api") as mock:
            app._evaluate(short, "gateway-test-key")
            mock.assert_not_called()
        with contextlib.closing(app.connect(short.db)) as db:
            data = json.loads(db.execute("SELECT evaluation FROM results WHERE profile=? LIMIT 1", (app.active_profile(db),)).fetchone()[0])
            self.assertIn("reused_from_profile", data)
            self.assertEqual(data["visual_faithfulness"]["score"], 5)
        short.force, short.limit = True, 1
        with patch.object(app, "call_api", return_value=concise_response()) as mock:
            app._evaluate(short, "gateway-test-key")
            self.assertEqual(mock.call_count, 1)

    def test_concise_filters_thumbnails_and_different_models_are_not_reused(self):
        args = self.a6_args("concise")
        with patch.object(app, "call_api", side_effect=[concise_response("不忠实：颜色不符。", visual_conflict=color_conflict()),
                concise_response(logic="不确定：推断依据不足。"), concise_response(), concise_response()]):
            app._evaluate(args, "gateway-test-key")
        with contextlib.closing(app.connect(args.db)) as db:
            rows = app.list_samples(db, {"visual_judgment": "不忠实"})
            self.assertEqual(rows["total"], 1)
            self.assertEqual(rows["items"][0]["view_count"], 1)
            self.assertTrue(rows["items"][0]["thumbnail_url"].endswith("?size=360"))
            self.assertEqual(app.list_samples(db, {"logic_judgment": "不确定"})["total"], 1)
            self.assertEqual(app.list_samples(db, {"status": "fail"})["total"], 1)
        args.model = "another-vision-model"
        with patch.object(app, "call_api", return_value=concise_response()) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 4)

    def test_a6_defaults_endpoint_and_key_isolation(self):
        args = self.a6_args()
        self.assertEqual(args.provider, "a6api")
        self.assertEqual(args.model, "qwen3.8-max")
        self.assertEqual(args.endpoint, "https://api.a6api.vip/v1/chat/completions")
        self.assertEqual(args.thinking, "omit")
        self.assertEqual(app.parse_args(["serve"]).output_dir.name, "cot_evaluation_a6api_qwen38max")
        self.assertEqual(app.parse_args(["serve", "--provider", "zhipu"]).output_dir.name, "cot_evaluation_glm46v")
        self.assertEqual(app.parse_args(["evaluate", "--base-url", "https://api.a6api.vip/v1/"]).endpoint, args.endpoint)
        with patch.dict(app.os.environ, {"ZHIPU_API_KEY": "legacy-only-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "A6API_KEY"):
                app.get_api_key(args)
        with patch.dict(app.os.environ, {"A6API_KEY": "  gateway-test-key  ", "ZHIPU_API_KEY": "legacy-only-key"}, clear=True):
            self.assertEqual(app.get_api_key(args), "gateway-test-key")
            self.assertEqual(app.get_api_key(self.args), "legacy-only-key")

    def test_a6_request_preserves_image_mime_and_omits_glm_options(self):
        args = self.a6_args()
        with contextlib.closing(app.connect(args.db)) as db:
            _, profile = app.get_profile(args, db)
            _, sample, trace = app.STORE.sample(db, 1)
        jpeg = self.root / "query.jpg"
        Image.new("RGB", (64, 48), "green").save(jpeg, "JPEG")
        sample["query_images"][0]["path"] = str(jpeg)
        payload = app.make_request(sample, trace, profile)
        self.assertEqual(payload["model"], "qwen3.8-max")
        self.assertNotIn("thinking", payload)
        self.assertNotIn("enable_thinking", payload)
        self.assertNotIn("response_format", payload)
        images = [c["image_url"]["url"] for c in payload["messages"][1]["content"] if c["type"] == "image_url"]
        self.assertTrue(images[0].startswith("data:image/jpeg;base64,"))
        self.assertTrue(images[1].startswith("data:image/png;base64,"))
        decoded = Image.open(io.BytesIO(base64.b64decode(images[0].split(",", 1)[1])))
        self.assertEqual(decoded.format, "JPEG")
        self.assertNotIn(str(self.root), json.dumps(payload))
        self.assertNotIn(sample["id"], json.dumps(payload))

    def test_a6_profiles_do_not_reuse_glm_results_and_resume_normally(self):
        with patch.object(app, "call_api", return_value=response()):
            app._evaluate(self.args, "offline-key")
        with contextlib.closing(app.connect(self.args.db)) as db:
            legacy = app.active_profile(db)
            # This is the exact original profile shape, so old hashes stay valid.
            config = json.loads(db.execute("SELECT config FROM profiles WHERE hash=?", (legacy,)).fetchone()[0])
            self.assertNotIn("provider", config)
            self.assertEqual(config["endpoint"], app.PROVIDERS["zhipu"]["endpoint"])
        args = self.a6_args()
        with patch.object(app, "call_api", return_value=response()) as mock:
            app._evaluate(args, "gateway-test-key")
            self.assertEqual(mock.call_count, 4)
            self.assertNotIn("thinking", mock.call_args.args[2])
        with contextlib.closing(app.connect(args.db)) as db:
            self.assertNotEqual(app.active_profile(db), legacy)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results").fetchone()[0], 8)
            self.assertEqual(app.summary(db)["judge"]["model"], "qwen3.8-max")
            self.assertEqual(app.summary(db)["judge"]["provider"], "a6api")
        with patch.object(app, "call_api") as mock:
            app._evaluate(args, "gateway-test-key")
            mock.assert_not_called()

    def test_a6_http_auth_response_id_and_non_json_error(self):
        body = io.BytesIO(json.dumps(response()).encode())
        body.headers = {"X-Oneapi-Request-Id": "offline-response-id"}
        with patch.object(app.urllib.request, "urlopen", return_value=body) as mock:
            value = app.call_api(app.DEFAULT_ENDPOINT, "gateway-test-key", {"model": "qwen3.8-max"}, 12)
            request = mock.call_args.args[0]
            self.assertEqual(request.full_url, app.DEFAULT_ENDPOINT)
            self.assertEqual(request.get_header("Authorization"), "Bearer gateway-test-key")
            self.assertEqual(json.loads(request.data)["model"], "qwen3.8-max")
            self.assertEqual(value["gateway_request_id"], "offline-response-id")
        with patch.object(app.urllib.request, "urlopen", return_value=io.BytesIO(b"<html>not an API response</html>")):
            with self.assertRaises(app.APIError) as caught:
                app.call_api(app.DEFAULT_ENDPOINT, "gateway-test-key", {}, 12)
            self.assertTrue(caught.exception.fatal)

    def test_a6_gateway_errors_quota_price_guard_and_rate_limits(self):
        for status, code, fatal, retryable in ((401, "invalid_api_key", True, False),
                (403, "insufficient_user_quota", True, False), (403, "token_quota_exhausted", True, False),
                (403, "model_not_allowed", True, False), (402, "marketplace_price_guard_triggered", True, False),
                (429, "insufficient_quota", True, False), (429, "user_concurrency_limit", False, True),
                (429, "model_request_rate_limit", False, True), (503, "upstream_timeout", False, True)):
            body = io.BytesIO(json.dumps({"error": {"code": code, "message": "gateway-test-key"}}).encode())
            error = urllib.error.HTTPError(app.DEFAULT_ENDPOINT, status, "test", {"Retry-After": "10", "X-Oneapi-Request-Id": "offline-error-id"}, body)
            with patch.object(app.urllib.request, "urlopen", side_effect=error):
                with self.assertRaises(app.APIError) as caught:
                    app.call_api(app.DEFAULT_ENDPOINT, "gateway-test-key", {}, 12)
            self.assertEqual(caught.exception.fatal, fatal)
            self.assertEqual(caught.exception.retryable, retryable)
            self.assertEqual(caught.exception.rate_limited, retryable and status == 429)
            self.assertNotIn("gateway-test-key", str(caught.exception))
            self.assertIn("offline-error-id", str(caught.exception))

    def test_two_image_capability_probe_and_failure(self):
        args = self.a6_args()
        payload, expected = app.make_vision_probe(args.model)
        images = [c["image_url"]["url"] for c in payload["messages"][0]["content"] if c["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        colors = {"red": (255,0,0), "green": (0,192,0), "blue": (0,0,255), "yellow": (255,255,0),
                  "magenta": (255,0,255), "cyan": (0,255,255)}
        for url, pair in zip(images, expected):
            im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            self.assertEqual([im.getpixel((60,60)), im.getpixel((180,60))], [colors[c] for c in pair])
        good = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"images": expected})}}]}
        with patch.dict(app.os.environ, {"A6API_KEY": "gateway-test-key"}, clear=True), patch.object(app, "make_vision_probe", return_value=(payload, expected)), patch.object(app, "call_api", return_value=good):
            app.check_api(args)
            self.assertTrue(json.loads((args.output_dir / "api_check.json").read_text())["passed"])
        bad = {"choices": [{"finish_reason": "stop", "message": {"content": '{"images": []}'}}]}
        with patch.dict(app.os.environ, {"A6API_KEY": "gateway-test-key"}, clear=True), patch.object(app, "call_api", return_value=bad):
            with self.assertRaisesRegex(ValueError, "Image capability check failed"):
                app.check_api(args)
            self.assertFalse(json.loads((args.output_dir / "api_check.json").read_text())["passed"])

    def test_schema_rejects_truncation_bad_scores_and_invented_quotes(self):
        with contextlib.closing(app.connect(self.args.db)) as db:
            row, sample, trace = app.STORE.sample(db, 1)
        self.assertEqual(app.parse_evaluation(response(), trace, sample)["verdict"], "pass")
        self.assertEqual(app.parse_evaluation(response(3), trace, sample)["verdict"], "review")
        self.assertEqual(app.parse_evaluation(response(2), trace, sample)["verdict"], "fail")
        bad = response(); bad["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError): app.parse_evaluation(bad, trace, sample)
        for bad_score in (True, "5", 6, 0):
            with self.assertRaises(ValueError): app.parse_evaluation(response(bad_score), trace, sample)
        bad = response(); content = json.loads(bad["choices"][0]["message"]["content"])
        content["issues"] = [{"dimension": "visual_faithfulness", "severity": "major", "quote": "invented quotation", "view_ids": [], "explanation": "bad"}]
        bad["choices"][0]["message"]["content"] = json.dumps(content)
        with self.assertRaises(ValueError): app.parse_evaluation(bad, trace, sample)
        content["issues"][0]["quote"] = "green surface"
        bad["choices"][0]["message"]["content"] = json.dumps(content)
        self.assertEqual(app.parse_evaluation(bad, trace, sample)["verdict"], "review")
        content["issues"][0]["view_ids"] = ["hidden_view"]
        bad["choices"][0]["message"]["content"] = json.dumps(content)
        with self.assertRaises(ValueError): app.parse_evaluation(bad, trace, sample)

    def test_evaluate_retry_resume_and_stale_source(self):
        invalid = response(); invalid["choices"][0]["message"]["content"] = "invalid JSON"
        with patch.object(app, "call_api", side_effect=[invalid, response(), response(), response(), response()]) as mock:
            app._evaluate(self.args, "offline-key")
            self.assertEqual(mock.call_count, 5)
        with patch.object(app, "call_api") as mock:
            app._evaluate(self.args, "offline-key")
            mock.assert_not_called()
        with contextlib.closing(app.connect(self.args.db)) as db:
            self.assertEqual(sum(r["done"] for r in app.summary(db)["datasets"]), 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 5)
        exported = self.args.output_dir / "evaluations_mvtec_ad.jsonl"
        record = json.loads(exported.read_text())
        self.assertEqual(record["evaluation"]["visual_faithfulness"]["score"], 5)
        self.assertNotIn("image_url", exported.read_text())
        path = self.traces / "traces_mvtec_ad.json"
        value = json.loads(path.read_text()); value["records"][-1]["reasoning"] += " "
        path.write_text(json.dumps(value)); app.prepare(self.args)
        with contextlib.closing(app.connect(self.args.db)) as db:
            self.assertEqual(sum(r["done"] for r in app.summary(db)["datasets"]), 3)
        with patch.object(app, "call_api", return_value=response()) as mock:
            app._evaluate(self.args, "offline-key")
            self.assertEqual(mock.call_count, 1)

    def test_auth_failure_stops_before_all_samples(self):
        with patch.object(app, "call_api", side_effect=app.APIError("HTTP 401", fatal=True)) as mock:
            with self.assertRaises(ValueError): app._evaluate(self.args, "offline-key")
            self.assertEqual(mock.call_count, 1)

    def test_api_http_errors_distinguish_transient_and_fatal(self):
        for status, code, fatal, retryable in ((401, "1004", True, False), (400, "1210", True, False),
                (429, "1302", False, True), (429, "1305", False, True), (400, "1302", False, True),
                (500, "500", False, True), (400, "1301", False, False), (429, "1113", True, False),
                (429, "1308", True, False), (429, "1310", True, False)):
            body = io.BytesIO(json.dumps({"error": {"code": code, "message": "offline-key"}}).encode())
            error = urllib.error.HTTPError(app.DEFAULT_ENDPOINT, status, "test", {"Retry-After": "2"}, body)
            with patch.object(app.urllib.request, "urlopen", side_effect=error):
                with self.assertRaises(app.APIError) as caught:
                    app.call_api(app.DEFAULT_ENDPOINT, "offline-key", {}, 1)
            self.assertEqual(caught.exception.fatal, fatal)
            self.assertEqual(caught.exception.retryable, retryable)
            self.assertNotIn("offline-key", str(caught.exception))
            self.assertEqual(caught.exception.rate_limited, retryable and code in {"1302", "1305"})

    def test_retry_after_seconds_and_http_date(self):
        self.assertEqual(app.retry_after_seconds("450"), 450)
        self.assertEqual(app.retry_after_seconds("-1"), 0)
        self.assertEqual(app.retry_after_seconds("invalid"), 0)
        self.assertEqual(app.retry_after_seconds("nan"), 0)
        stamp = app.dt.datetime(2015, 10, 21, 7, 28, tzinfo=app.dt.timezone.utc).timestamp()
        with patch.object(app.time, "time", return_value=stamp - 90):
            self.assertEqual(app.retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT"), 90)

    def test_shared_cooldown_rechecks_workers_already_waiting(self):
        clock = [100.0]
        delays = []
        stopped = [False]
        limiter = None

        class FakeStop:
            def is_set(self):
                return stopped[0]

            def wait(self, delay):
                delays.append(delay)
                clock[0] += delay
                if len(delays) == 1:
                    # Another worker gets 429 while this worker is already asleep.
                    limiter.throttle()
                return stopped[0]

        with patch.object(app.time, "monotonic", side_effect=lambda: clock[0]), patch.object(app.random, "random", return_value=0):
            limiter = app.RateLimiter(5, FakeStop(), rate_limit_wait=30)
            self.assertTrue(limiter.wait())
            self.assertTrue(limiter.wait())
            self.assertEqual(delays, [5, 30])
            self.assertEqual(clock[0], 135)
            self.assertEqual(limiter.throttle(), 60)
            self.assertEqual(limiter.throttle(), 120)
            self.assertEqual(limiter.throttle(), 240)
            self.assertEqual(limiter.throttle(), 300)
            self.assertEqual(limiter.throttle(retry_after=450), 450)
            limiter.success()
            clock[0] += 450
            self.assertEqual(limiter.throttle(), 30)
            stopped[0] = True
            self.assertFalse(limiter.wait())

    def test_exhausted_overload_still_cools_down_next_sample(self):
        self.args.limit, self.args.retries = 2, 0
        limiter = Mock()
        limiter.wait.return_value = True
        limiter.throttle.return_value = 30
        error = app.APIError("HTTP 429 / 1305", retryable=True, rate_limited=True)
        with patch.object(app, "RateLimiter", return_value=limiter), patch.object(app, "call_api", side_effect=[error, response()]) as mock:
            with self.assertRaisesRegex(ValueError, "1 samples failed"):
                app._evaluate(self.args, "offline-key")
            self.assertEqual(mock.call_count, 2)
            limiter.throttle.assert_called_once_with(0)
            self.assertEqual(limiter.wait.call_count, 2)
            limiter.success.assert_called_once()
        with contextlib.closing(app.connect(self.args.db)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE status='done'").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE status='error'").fetchone()[0], 1)

    def test_retry_controls_do_not_invalidate_successful_profile(self):
        with contextlib.closing(app.connect(self.args.db)) as db:
            first, _ = app.get_profile(self.args, db)
            self.args.rate_limit_wait = 60
            self.args.retries = 8
            self.args.workers = 2
            self.args.request_interval = 3
            second, _ = app.get_profile(self.args, db)
            self.assertEqual(first, second)

    def test_multiview_pairing_and_gt_overlay(self):
        ds = "mvtec3d"
        path = self.manifests / f"{ds}.json"
        value = json.loads(path.read_text()); sample = value["samples"][0]
        sample["query_images"].append({**sample["query_images"][0], "view_id": "view_2"})
        sample["normal_reference"]["images"].append({**sample["normal_reference"]["images"][0], "view_id": "view_2"})
        sample["is_anomaly"] = True
        sample["supervision"]["bboxes"] = [{"view_id": "view_2", "xyxy_pixel": [10, 12, 30, 35]},
                                           {"view_id": "view_2", "xyxy_pixel": [45, 15, 55, 25]}]
        path.write_text(json.dumps(value))
        unboxed_path = self.manifests / "mvtec_ad.json"
        unboxed = json.loads(unboxed_path.read_text())
        unboxed["samples"][0]["is_anomaly"] = True
        unboxed_path.write_text(json.dumps(unboxed))
        app.prepare(self.args)
        with contextlib.closing(app.connect(self.args.db)) as db:
            _, profile = app.get_profile(self.args, db)
            sid = db.execute("SELECT id FROM samples WHERE dataset=?", (ds,)).fetchone()[0]
            _, sample, trace = app.STORE.sample(db, sid)
            payload = app.make_request(sample, trace, profile)
            self.assertEqual([c["text"].split(";")[0] for c in payload["messages"][1]["content"] if c["type"] == "text"][:4],
                             ["query view_id=query", "reference view_id=query", "query view_id=view_2", "reference view_id=view_2"])
            unmarked = Image.open(io.BytesIO(app.overlay(db, sid, 0)))
            marked = Image.open(io.BytesIO(app.overlay(db, sid, 1)))
            self.assertEqual(marked.getpixel((10,12)), (240,68,91))
            self.assertNotEqual(unmarked.getpixel((10,12)), (240,68,91))

        args = self.a6_args("concise")
        self.assertEqual(app.parse_args(["evaluate"]).output_dir.name, "cot_evaluation_a6api_qwen38max")
        args.batch_size, args.batch_max_images = 5, 10
        entries = [{"sample_index": i, "visual_faithfulness": "不忠实：颜色不符。",
                    "reasoning_consistency": "合理：推导一致。", "visual_conflict": color_conflict()}
                   for i in range(3)]
        with patch.object(app, "call_api", side_effect=[batch_response(3, entries), batch_response(1)]) as mock:
            app._evaluate(args, "offline-key")
            self.assertEqual(mock.call_count, 2)
        payload = mock.call_args_list[0].args[2]
        content = payload["messages"][1]["content"]
        start = content.index({"type": "text", "text": "SAMPLE_BEGIN sample_index=1"})
        end = content.index({"type": "text", "text": "SAMPLE_END sample_index=1"})
        self.assertEqual([c["text"].split(";")[0] for c in content[start + 1:end] if c["type"] == "text"][:5],
                         ["query view_id=query", "reference view_id=query", "query view_id=view_2",
                          "annotation_cue view_id=view_2", "reference view_id=view_2"])
        images = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
        self.assertEqual(len(images), 9)
        self.assertEqual(sum(c.get("text", "").startswith("annotation_cue") for c in content), 1)
        with Image.open(io.BytesIO(base64.b64decode(images[5].split(",", 1)[1]))) as image:
            self.assertEqual(image.getpixel((10, 12)), (240, 68, 91))
            self.assertEqual(image.getpixel((45, 15)), (240, 68, 91))
        with contextlib.closing(app.connect(args.db)) as db:
            evaluations = {row["dataset"]: json.loads(row["evaluation"]) for row in db.execute("""
                SELECT s.dataset,r.evaluation FROM results r JOIN samples s ON s.id=r.sample_id
                WHERE r.profile=?
            """, (app.active_profile(db),))}
            self.assertTrue(evaluations[ds]["region_review"])
            self.assertEqual(evaluations[ds]["judgments"]["visual_faithfulness"], "不确定")
            for dataset in ("anomaly_shapenet", "mvtec_ad"):
                self.assertNotIn("region_review", evaluations[dataset])
                self.assertEqual(evaluations[dataset]["judgments"]["visual_faithfulness"], "不忠实")

        scored = self.a6_args("scored")
        scored.image_id = sample["id"]
        with patch.object(app, "call_api", return_value=response()):
            app._evaluate(scored, "offline-key")
        boxed = concise_response("不忠实：颜色不符。", visual_conflict={**color_conflict(), "query_view_id": "view_2"})
        args.batch_size, args.max_tokens, args.image_id = 1, 513, sample["id"]
        with patch.object(app, "call_api", return_value=boxed) as mock:
            app._evaluate(args, "offline-key")
            mock.assert_called_once()
        with contextlib.closing(app.connect(args.db)) as db:
            evaluation = json.loads(db.execute("SELECT evaluation FROM results WHERE sample_id=? AND profile=?",
                                                (sid, app.active_profile(db))).fetchone()[0])
            self.assertEqual(evaluation["judgments"]["visual_faithfulness"], "不忠实")

    def test_filters_pagination_and_profile_change(self):
        with patch.object(app, "call_api", return_value=response()): app._evaluate(self.args, "offline-key")
        with contextlib.closing(app.connect(self.args.db)) as db:
            page = app.list_samples(db, {"page_size": "1", "page": "2", "status": "pass"})
            self.assertEqual(page["total"], 4); self.assertEqual(len(page["items"]), 1); self.assertEqual(page["page"], 2)
            self.assertEqual(app.list_samples(db, {"dataset": "mvtec_ad", "max_visual": "3"})["total"], 0)
            self.assertEqual(app.list_samples(db, {"dataset": "mvtec_ad", "q": "sample"})["total"], 1)
            self.args.temperature = 0.3; app.get_profile(self.args, db)
            self.assertEqual(app.list_samples(db, {"status": "pending"})["total"], 4)
        self.assertEqual(app.split_trace("<think>x</think><locate>[1,2,3,4]</locate><answer>Yes</answer>")["location"], "[1,2,3,4]")

    def test_http_images_overlay_review_and_path_restriction(self):
        server = app.make_server(self.args)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def get(path):
            with opener.open(base+path) as r: return r.read()
        try:
            self.assertIn(b"TraceLens", get("/"))
            page = json.loads(get("/api/samples?page_size=2")); self.assertEqual(page["total"], 4)
            data = json.loads(get("/api/sample/1")); self.assertIn("think", data["tags"])
            self.assertEqual(Image.open(io.BytesIO(get(data["images"]["query"][0]["url"]))).size, (64,48))
            thumb = Image.open(io.BytesIO(get(data["images"]["query"][0]["url"] + "?size=32")))
            self.assertEqual(thumb.size, (32,24))
            self.assertEqual(thumb.format, "JPEG")
            self.assertIn(b"sample-grid", get("/gallery.css"))
            self.assertEqual(Image.open(io.BytesIO(get("/api/overlay/1/0"))).size, (64,48))
            request = urllib.request.Request(base+"/api/review/1", data=b'{"decision":"flagged","note":"check color"}', headers={"Content-Type": "application/json"})
            with opener.open(request) as r: self.assertTrue(json.load(r)["saved"])
            self.assertEqual(json.loads(get("/api/sample/1"))["review"]["note"], "check color")
            request = urllib.request.Request(base+"/api/review/1", data=b'{"decision":"checked"}', headers={"Content-Type": "application/json", "Origin": "http://elsewhere.example"})
            with self.assertRaises(urllib.error.HTTPError) as exc: opener.open(request)
            self.assertEqual(exc.exception.code, 403)
            for path in ("/api/image/99999", "/api/image?path=/etc/passwd", "/../evaluate_cot_glm.py"):
                with self.assertRaises(urllib.error.HTTPError) as exc: get(path)
                self.assertEqual(exc.exception.code, 404)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()
