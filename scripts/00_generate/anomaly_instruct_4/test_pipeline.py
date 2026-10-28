import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from build_mvtec3d_bboxes import draw_bbox_prompt
from dataset_utils import canonical_type, mask_bbox
from generate_traces_qwen3vl import format_user_prompt, teacher_image_paths
from validate_trace import bbox_iou, parse_locations, validate_reasoning


def anomaly_sample(modality="2d"):
    query_images = [{"path": "/q.png", "view_id": "query", "width": 400, "height": 300}]
    boxes = [{"view_id": "query", "xyxy_pixel": [101, 52, 203, 177]}]
    if modality != "2d":
        query_images = [
            {"path": "/v0.png", "view_id": "view_000", "width": 800, "height": 800},
            {"path": "/v1.png", "view_id": "view_017", "width": 800, "height": 800},
        ]
        boxes = [{"view_id": "view_017", "xyxy_pixel": [101, 52, 203, 177]}]
        bbox_prompt_images = [
            {"path": "/b0.png", "view_id": "view_000"},
            {"path": "/b1.png", "view_id": "view_017"},
        ]
        reference_images = [
            {"path": "/r0.png", "view_id": "view_000"},
            {"path": "/r1.png", "view_id": "view_017"},
        ]
    else:
        bbox_prompt_images = [{"path": "/b.png", "view_id": "query"}]
        reference_images = [{"path": "/r.png", "view_id": "query"}]
    return {
        "id": "sample_1",
        "dataset": "test",
        "target_modality": modality,
        "object_category": "object",
        "is_anomaly": True,
        "query_images": query_images,
        "normal_reference": {"images": reference_images},
        "supervision": {
            "bbox_coordinates_exposed_to_teacher": True,
            "mask_paths": [{"path": "/mask.png"}],
            "bbox_prompt_images": bbox_prompt_images,
            "bboxes": boxes,
            "anomaly_type": {"canonical": "concavity"},
        },
    }


class PipelineTests(unittest.TestCase):
    def test_mask_bbox_and_prompt_are_written_outside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query = root / "query.png"
            mask = root / "mask.png"
            output = root / "derived" / "bbox.png"
            Image.new("RGB", (64, 48), "white").save(query)
            mask_image = Image.new("L", (64, 48), 0)
            ImageDraw.Draw(mask_image).rectangle((10, 11, 30, 35), fill=255)
            mask_image.save(mask)
            self.assertEqual(mask_bbox(mask)[0], [10, 11, 30, 35])
            self.assertEqual(draw_bbox_prompt(query, mask, output), [10, 11, 30, 35])
            self.assertTrue(output.exists())

    def test_empty_multiview_mask_creates_unboxed_context_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query, mask, output = root / "q.png", root / "m.png", root / "o.png"
            Image.new("RGB", (32, 32), "white").save(query)
            Image.new("L", (32, 32), 0).save(mask)
            self.assertIsNone(draw_bbox_prompt(query, mask, output))
            self.assertTrue(output.exists())

    def test_prompt_contains_exact_gt_bbox_numbers(self):
        prompt = format_user_prompt(anomaly_sample())
        self.assertIn("[101,52,203,177]", prompt)
        self.assertNotIn("copy exactly", prompt.lower())
        self.assertIn("teacher hint", prompt)
        self.assertIn("focus on each visibly boxed region", prompt)
        self.assertNotIn("For multiview samples", prompt)
        self.assertIn("Required response branch: ANOMALOUS_2D", prompt)

    def test_multiview_prompt_has_explicit_image_role_mapping(self):
        prompt = format_user_prompt(anomaly_sample("3d_multiview"))
        self.assertIn("Image 1: original query (view_id=view_000)", prompt)
        self.assertIn("Image 3: localization cue (view_id=view_000)", prompt)
        self.assertIn("Image 6: same-category normal reference (view_id=view_017)", prompt)
        self.assertIn("match images by view_id", prompt)
        self.assertIn("include every bbox-marked view exactly once", prompt)
        self.assertIn("Required response branch: ANOMALOUS_MULTIVIEW", prompt)

    def test_normal_prompt_selects_normal_branch_without_bbox_cue(self):
        sample = anomaly_sample()
        sample["is_anomaly"] = False
        prompt = format_user_prompt(sample)
        self.assertIn("Required response branch: NORMAL", prompt)
        self.assertNotIn("localization cue", prompt)
        self.assertNotIn("focus on each visibly boxed region", prompt)

    def test_source_mask_is_never_a_teacher_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = anomaly_sample()
            for name in ("q.png", "b.png", "r.png", "mask.png"):
                Image.new("RGB", (8, 8), "white").save(root / name)
            sample["query_images"][0]["path"] = str(root / "q.png")
            sample["supervision"]["bbox_prompt_images"][0]["path"] = str(root / "b.png")
            sample["supervision"]["mask_paths"][0]["path"] = str(root / "mask.png")
            sample["normal_reference"]["images"][0]["path"] = str(root / "r.png")
            paths = teacher_image_paths(sample)
            self.assertNotIn(str(root / "mask.png"), paths)
            self.assertEqual(paths, [str(root / "q.png"), str(root / "b.png"), str(root / "r.png")])

    def test_2d_and_multiview_location_validation(self):
        think = " ".join(["visible"] * 90)
        sample = anomaly_sample()
        reasoning = (
            f"<think>{think}</think><location>[101,52,203,177]</location>"
            "<type>concavity</type><answer>Yes</answer>"
        )
        self.assertTrue(validate_reasoning(reasoning, sample).valid)

        high_iou = (
            f"<think>{think}</think><location>[101,52,203,176]</location>"
            "<type>concavity</type><answer>Yes</answer>"
        )
        self.assertTrue(validate_reasoning(high_iou, sample, min_iou=0.9).valid)

        low_iou = (
            f"<think>{think}</think><location>[101,52,203,160]</location>"
            "<type>concavity</type><answer>Yes</answer>"
        )
        self.assertFalse(validate_reasoning(low_iou, sample, min_iou=0.9).valid)

        sample = anomaly_sample("3d_multiview")
        reasoning = (
            f"<think>{think}</think>"
            '<location>[{"view_id":"view_017","bbox":[101,52,203,177]}]</location>'
            "<type>concavity</type><answer>Yes</answer>"
        )
        self.assertTrue(validate_reasoning(reasoning, sample).valid)

    def test_location_parser_and_iou(self):
        self.assertEqual(parse_locations("[1,2,3,4]", "2d")[0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(bbox_iou([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)

    def test_conversation_type_normalization(self):
        self.assertEqual(canonical_type("There is a notable concavity on the surface."), "concavity")
        self.assertEqual(canonical_type("A darker irregularly shaped spot is visible."), "dark spot")


if __name__ == "__main__":
    unittest.main()
