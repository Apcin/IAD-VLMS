"""CPU-only unit tests for the flag-gated reward controls in reward.py
(REPORT2.md §2.6 redesign, band b=0.1; §2.7 ablation-arm component switches).

Self-contained: runs under pytest or as plain python. The embedding-similarity
server call (NomicEmbeddingModel.similarity) is monkeypatched with a
deterministic table, so no network and no GPU are needed.

Battery: 12 synthetic completion/solution pairs covering correct-yes with
good/bad tags, the hedging exploit (wrong 'no' verdict with perfect tags),
correct-no, malformed tags, and missing answers.

Frozen baseline: the ACC_BASELINE / FMT_BASELINE arrays below are hardcoded
expected values derived by hand from the SHIPPED code paths of
accuracy_reward / consistency_reward (staircase thresholds .90/.80/.70/.55/.40
-> {1.0,.9,.7,.5,.2,0}; location 3x3 grid match; (type+loc)/2 + 1[ans==gt];
format fullmatch patterns). Test (a) asserts that with ALL flags unset the
module reproduces them exactly — i.e. every addition is a no-op by default.
"""

import math
import os
import sys
from contextlib import contextmanager

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from reward_process import type_reward  # noqa: E402

# ── Monkeypatch the embedding-similarity server call (no network) ────────────
# Keyed on the normalized (pred, gt) strings that reach NomicEmbeddingModel.
# The raw values are chosen so staircase and raw-similarity paths diverge
# (0.75 -> staircase 0.7) and so an exact-1.0 similarity exists for the
# band-form ceiling check.
SIM_TABLE = {
    ("crack", "crack"): 1.0,
    ("scratch", "scratch"): 0.95,
    ("scrape", "scratch"): 0.75,
    ("discoloration", "scratch"): 0.30,
}


def _fake_similarity(self, text1, text2):
    return SIM_TABLE.get((text1, text2), SIM_TABLE.get((text2, text1), 0.0))


type_reward.NomicEmbeddingModel.similarity = _fake_similarity

import reward  # noqa: E402  (imported after the patch; flags parse at call time anyway)

# ── Env helpers ──────────────────────────────────────────────────────────────
_FLAG_ENVS = ("REWARD_GATE_TAGS", "REWARD_BAND_FORM", "REWARD_COMPONENTS")


@contextmanager
def flags(**env):
    """Set exactly the given reward-control envs; clear the rest; restore after."""
    saved = {k: os.environ.get(k) for k in _FLAG_ENVS}
    try:
        for k in _FLAG_ENVS:
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def close(a, b):
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)


def assert_list(got, want, label):
    assert len(got) == len(want), f"{label}: length {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
        assert close(g, w), f"{label}: case {i + 1}: got {g!r}, want {w!r}"


# ── The 12-case battery ──────────────────────────────────────────────────────
THINK = "<think>The image shows a clear surface defect pattern here.</think>"

SOL_YES_CRACK = "<answer>yes</answer><type>crack</type><location>top left</location>"
SOL_YES_SCRATCH = "<answer>yes</answer><type>scratch</type><location>top left</location>"
SOL_NO = "<answer>no</answer>"

CASES = [
    # 1  correct-yes, perfect tags (raw sim 1.0 -> band ceiling case)
    (THINK + "<location>top left</location><type>crack</type><answer>yes</answer>", SOL_YES_CRACK),
    # 2  correct-yes, good-but-inexact type (raw .75 / staircase .7), loc correct
    (THINK + "<location>top left</location><type>scrape</type><answer>yes</answer>", SOL_YES_SCRATCH),
    # 3  correct-yes, bad tags (raw .30 / staircase 0, loc wrong)
    (THINK + "<location>bottom right</location><type>discoloration</type><answer>yes</answer>", SOL_YES_SCRATCH),
    # 4  HEDGING EXPLOIT: gt=yes, verdict 'no', perfect tags
    (THINK + "<location>top left</location><type>crack</type><answer>no</answer>", SOL_YES_CRACK),
    # 5  correct-no, clean
    (THINK + "<answer>no</answer>", SOL_NO),
    # 6  wrong-yes on gt=no, with tags
    (THINK + "<location>top left</location><type>crack</type><answer>yes</answer>", SOL_NO),
    # 7  malformed type tag (no closing '</type>'), loc + answer correct
    (THINK + "<location>top left</location><type>scratch</type<answer>yes</answer>", SOL_YES_SCRATCH),
    # 8  missing answer, perfect tags (hedge-by-omission)
    (THINK + "<location>top left</location><type>crack</type>", SOL_YES_CRACK),
    # 9  correct-yes, no tags at all
    (THINK + "<answer>yes</answer>", SOL_YES_CRACK),
    # 10 honest wrong-no on gt=yes, no tags
    (THINK + "<answer>no</answer>", SOL_YES_CRACK),
    # 11 correct-no with a stray type tag
    (THINK + "<type>crack</type><answer>no</answer>", SOL_NO),
    # 12 missing answer on gt=no
    (THINK, SOL_NO),
]

COMPLETIONS = [[{"content": c}] for c, _ in CASES]
SOLUTIONS = [s for _, s in CASES]

# ── Frozen baselines (hand-derived from the SHIPPED code paths) ──────────────
# accuracy_reward: gt=no -> 1[ans=='no']; gt=yes -> (stair(type)+loc)/2 + 1[ans=='yes']
ACC_BASELINE = [2.0, 1.85, 1.0, 1.0, 1.0, 0.0, 1.5, 1.0, 1.0, 0.0, 1.0, 0.0]
# consistency_reward: fullmatch pattern_yes / pattern_no (answer text not inspected)
FMT_BASELINE = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

IDX_EXPLOIT = 3        # case 4 (0-based)
IDX_OMIT_HEDGE = 7     # case 8

# ── Expected values under each flag configuration (hand-derived) ─────────────
ACC_GATED = [2.0, 1.85, 1.0, 0.0, 1.0, 0.0, 1.5, 0.0, 1.0, 0.0, 1.0, 0.0]
ACC_BAND = [1.05, 1.0375, 0.965, 0.0, 1.0, 0.0, 1.0, 0.0, 0.95, 0.0, 1.0, 0.0]
ACC_VERDICT_FORMAT = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
ACC_NO_TYPE = [1.5, 1.5, 1.0, 0.5, 1.0, 0.0, 1.5, 0.5, 1.0, 0.0, 1.0, 0.0]
ACC_NO_LOC = [1.5, 1.35, 1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.0]
ZEROS = [0.0] * 12


def _acc():
    return reward.accuracy_reward(COMPLETIONS, SOLUTIONS)


def _fmt():
    return reward.consistency_reward(COMPLETIONS, SOLUTIONS)


# ── (a) all flags unset => identical to the shipped behavior ─────────────────
def test_baseline_flags_unset():
    with flags():
        assert_list(_acc(), ACC_BASELINE, "baseline accuracy")
        assert_list(_fmt(), FMT_BASELINE, "baseline format")
        # the shipped exploit really pays out ungated (>0 answer-term credit)
        acc = _acc()
        assert acc[IDX_EXPLOIT] > 0.0, "exploit should earn credit when ungated"
        assert acc[IDX_OMIT_HEDGE] > 0.0, "omission hedge should earn credit when ungated"


def test_baseline_default_components_string_is_shipped_path():
    # REWARD_COMPONENTS set to exactly the default list must also be a no-op
    with flags(REWARD_COMPONENTS="verdict,format,type,loc"):
        assert_list(_acc(), ACC_BASELINE, "default-components accuracy")
        assert_list(_fmt(), FMT_BASELINE, "default-components format")


def test_flag_zero_values_are_shipped_path():
    with flags(REWARD_GATE_TAGS="0", REWARD_BAND_FORM="0"):
        assert_list(_acc(), ACC_BASELINE, "flags=0 accuracy")
        assert_list(_fmt(), FMT_BASELINE, "flags=0 format")


# ── (b) the verdict gate zeroes the hedging exploit ──────────────────────────
def test_gate_zeroes_exploit():
    with flags(REWARD_GATE_TAGS="1"):
        acc = _acc()
        assert_list(acc, ACC_GATED, "gated accuracy")
        assert close(acc[IDX_EXPLOIT], 0.0), "gate must zero the hedging exploit"
        assert close(acc[IDX_OMIT_HEDGE], 0.0), "gate must zero the omission hedge"
        # format term untouched by the gate
        assert_list(_fmt(), FMT_BASELINE, "gated format")


# ── (c) band form: values, implied gate, ceilings 2.0 (no) vs 2.05 (yes) ─────
def test_band_form_values_and_ceilings():
    with flags(REWARD_BAND_FORM="1"):
        acc = _acc()
        fmt = _fmt()
        assert_list(acc, ACC_BAND, "band accuracy")
        assert_list(fmt, FMT_BASELINE, "band format (must be unchanged)")
        # band implies the gate even with REWARD_GATE_TAGS unset
        assert close(acc[IDX_EXPLOIT], 0.0), "band form must gate the exploit"
        # raw-similarity bypass: case 2 uses raw .75 (1.0375), not staircase .7 (1.035)
        assert close(acc[1], 1.0375), "band must use RAW similarity, not the staircase"
        totals = [a + f for a, f in zip(acc, fmt)]
        yes_totals = [t for t, (_, s) in zip(totals, CASES) if "<answer>yes</answer>" in s]
        no_totals = [t for t, (_, s) in zip(totals, CASES) if "<answer>no</answer>" in s]
        assert close(max(yes_totals), 2.05), f"gt=yes ceiling must be 2.05, got {max(yes_totals)}"
        assert close(max(no_totals), 2.0), f"gt=no ceiling must be 2.0, got {max(no_totals)}"


def test_band_overrides_gate_and_components_for_answer_term():
    # Precedence: BAND_FORM wins over GATE and COMPONENTS for the answer term;
    # the format term still obeys its component switch.
    with flags(REWARD_BAND_FORM="1", REWARD_GATE_TAGS="1", REWARD_COMPONENTS="format"):
        assert_list(_acc(), ACC_BAND, "band+gate+components accuracy")
        assert_list(_fmt(), FMT_BASELINE, "format kept (format in components)")
    with flags(REWARD_BAND_FORM="1", REWARD_COMPONENTS="verdict"):
        assert_list(_acc(), ACC_BAND, "band accuracy with components=verdict")
        assert_list(_fmt(), ZEROS, "format zeroed (format omitted)")


# ── (d) component switches zero exactly their component ──────────────────────
def test_components_verdict_format():
    with flags(REWARD_COMPONENTS="verdict,format"):
        assert_list(_acc(), ACC_VERDICT_FORMAT, "verdict+format accuracy")
        assert_list(_fmt(), FMT_BASELINE, "verdict+format format")


def test_components_no_format():
    with flags(REWARD_COMPONENTS="verdict,type,loc"):
        assert_list(_acc(), ACC_BASELINE, "no-format accuracy (unchanged)")
        assert_list(_fmt(), ZEROS, "no-format format (zeroed)")


def test_components_no_type():
    with flags(REWARD_COMPONENTS="verdict,format,loc"):
        acc = _acc()
        assert_list(acc, ACC_NO_TYPE, "no-type accuracy")
        # the removed mass is exactly the type sub-signal (stair(type)/2)
        deltas = [b - a for b, a in zip(ACC_BASELINE, acc)]
        assert_list(deltas, [0.5, 0.35, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                    "no-type removed exactly the type component")


def test_components_no_loc():
    with flags(REWARD_COMPONENTS="verdict,format,type"):
        acc = _acc()
        assert_list(acc, ACC_NO_LOC, "no-loc accuracy")
        deltas = [b - a for b, a in zip(ACC_BASELINE, acc)]
        assert_list(deltas, [0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                    "no-loc removed exactly the loc component")


def test_components_no_verdict():
    # S6-style: verdict credit gone, ungated tag credit remains
    with flags(REWARD_COMPONENTS="format,type,loc"):
        acc = _acc()
        want = [1.0, 0.85, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]
        assert_list(acc, want, "no-verdict accuracy")
        assert_list(_fmt(), FMT_BASELINE, "no-verdict format")


def test_gate_composes_with_components():
    # gate + no-type: remaining loc credit still gated on the verdict
    with flags(REWARD_GATE_TAGS="1", REWARD_COMPONENTS="verdict,format,loc"):
        acc = _acc()
        want = [1.5, 1.5, 1.0, 0.0, 1.0, 0.0, 1.5, 0.0, 1.0, 0.0, 1.0, 0.0]
        assert_list(acc, want, "gate+no-type accuracy")


def test_call_time_parsing():
    # Same imported module, consecutive calls under different envs must differ:
    # proves flags are parsed at call time, not import time.
    with flags():
        base = _acc()
    with flags(REWARD_GATE_TAGS="1"):
        gated = _acc()
    with flags():
        back = _acc()
    assert not close(base[IDX_EXPLOIT], gated[IDX_EXPLOIT]), "env toggle had no effect"
    assert_list(back, base, "unset envs restore shipped behavior in-process")


ALL_TESTS = [
    test_baseline_flags_unset,
    test_baseline_default_components_string_is_shipped_path,
    test_flag_zero_values_are_shipped_path,
    test_gate_zeroes_exploit,
    test_band_form_values_and_ceilings,
    test_band_overrides_gate_and_components_for_answer_term,
    test_components_verdict_format,
    test_components_no_format,
    test_components_no_type,
    test_components_no_loc,
    test_components_no_verdict,
    test_gate_composes_with_components,
    test_call_time_parsing,
]


if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} tests passed")
    sys.exit(1 if failed else 0)
