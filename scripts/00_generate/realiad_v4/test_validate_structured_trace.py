import unittest

from validate_structured_trace import validate_reasoning


THINK = " ".join(["surface"] * 120)


class ValidateStructuredTraceTest(unittest.TestCase):
    def test_valid_normal(self):
        result = validate_reasoning(
            f"<think>{THINK}</think><answer>No</answer>",
            is_anomaly=False,
        )
        self.assertTrue(result.valid, result.errors)

    def test_valid_anomaly(self):
        result = validate_reasoning(
            f"<think>{THINK}</think>"
            "<location>middle-left</location><type>Scratch</type><answer>Yes</answer>",
            is_anomaly=True,
            expected_type="Scratch",
            expected_location="middle-left",
        )
        self.assertTrue(result.valid, result.errors)

    def test_rejects_label_mismatch_and_leakage(self):
        leaked = "ground truth " + " ".join(["surface"] * 118)
        result = validate_reasoning(
            f"<think>{leaked}</think>"
            "<location>top-right</location><type>Pit</type><answer>Yes</answer>",
            is_anomaly=True,
            expected_type="Scratch",
            expected_location="middle-left",
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("ground-truth" in error for error in result.errors))
        self.assertTrue(any("defect type" in error for error in result.errors))
        self.assertTrue(any("location disagrees" in error for error in result.errors))

    def test_rejects_normal_with_anomaly_tags(self):
        result = validate_reasoning(
            f"<think>{THINK}</think><type>Scratch</type><answer>No</answer>",
            is_anomaly=False,
        )
        self.assertFalse(result.valid)

    def test_rejects_extra_location_and_mask_leakage(self):
        leaked = "the mask " + " ".join(["surface"] * 118)
        result = validate_reasoning(
            f"<think>{leaked}</think>"
            "<location>middle-left, top-right</location>"
            "<type>Scratch</type><answer>Yes</answer>",
            is_anomaly=True,
            expected_type="Scratch",
            expected_location="middle-left",
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("mask leakage" in error for error in result.errors))
        self.assertTrue(any("location disagrees" in error for error in result.errors))

    def test_rejects_nested_tag_in_think(self):
        nested = " ".join(["surface"] * 119) + " <answer>No</answer>"
        result = validate_reasoning(
            f"<think>{nested}</think><answer>No</answer>",
            is_anomaly=False,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("nested XML" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
