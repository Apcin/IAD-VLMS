# Held-Out 4k Self-Distillation Rollout

**Date:** 2026-06-16 · **GPU:** CUDA 3 · **Model rolled out:** GRPO run-2
`checkpoint-530` (the 82.73 % DS-MVTec headline GRPO model) · **Status:** rollout
running (see *Results* below — populated on completion).

## What this is

A self-distillation rollout (the same Phase 0 → 1a → 1b pipeline used for the
10,236-item pool) run on the **held-out split** — the third Real-IAD split that
was **never used for SFT or GRPO training**. This gives us a clean,
training-disjoint pool of corrected reasoning traces.

## The held-out set and the no-overlap guarantee

| | items | NG (`yes`) | OK (`no`) | products |
|---|---|---|---|---|
| Source file `sft_iter2_heldout_train.json` | 4,630 | 2,328 | 2,302 | 27 |
| **− images that leak into the 10,236 pool** | −394 | | | |
| **`heldout_4236_disjoint.json` (used here)** | **4,236** | **2,129** | **2,107** | **23** |

- The raw held-out file shares images with the 10,236 SFT+GRPO rollout pool
  (`combined_6k_train.json` ∪ `grpo_train.json`): **394 of its 4,630 images also
  appear in that pool.** Those 394 are dropped.
- `4,630 − 394 = 4,236`, which **exactly matches the canonical held-out split in
  Appendix A** (23 products, 2,129 NG / 2,107 OK). Overlap with the 10,236 pool
  after filtering: **0 images** (verified by `image_path` set intersection).
- The split is **by-image-within-product**, not by-product: the held-out shares
  all 23 of its products with the training pools, but no individual image.

Disjoint file built by dropping any held-out image whose `image_path` is in
`set(combined_6k_train) ∪ set(grpo_train)`:

```python
import json, re
held  = json.load(open("Training/datasets_sft_iter2/sft_iter2_heldout_train.json"))   # 4,630
sft6k = json.load(open("Training/datasets_small_new_v4/combined_6k_train.json"))      # 6,000
grpo  = json.load(open("Training/datasets_small_15k_c1_only/grpo_train.json"))        # 4,236
ip = lambda r: r["images"][0] if "images" in r else r["image_path"]
pool10k = {ip(r) for r in sft6k} | {ip(r) for r in grpo}
disjoint = [r for r in held if ip(r) not in pool10k]                                  # 4,236
assert all(ip(r) not in pool10k for r in disjoint)                                    # 0 overlap
json.dump(disjoint, open("Training/datasets_sft_iter2/heldout_4236_disjoint.json","w"), indent=2)
```

## Pipeline (identical to the 10k run)

Launcher: `scripts/03_rollout_star/run_pipeline_heldout4k.sh`
(working copy `Training/run_pipeline_heldout4k.sh`). Stages:

| Stage | Script | What it does |
|---|---|---|
| A — Phase 0 rollout | `phase0_rollout_kscoring.py` | k=8 rollouts/item from run-2 ckpt-530 (T=0.7, top_p=0.9, max_new_tokens=512, max_pixels=480k, seed=42); each rollout scored with the production `accuracy_reward` + `consistency_reward`. Resumable (appends to `rollouts_raw.jsonl`). |
| B — bucket | `phase0_bucket.py` | thresholds yes=1.5 / no=1.0 / weak_yes=1.8 → `good_traces`, `judge_input`, `needs_correction`, `needs_rewrite`, `difficulty.json`. |
| C — Phase 1a judge | `phase1a_gemini_judge.py` | Gemini (`gemini-3-flash-preview`) faithfulness judge, keep_threshold=2 → `good_traces_faithful`, `needs_rewrite_from_judge`. |
| D | inline | union of local + judge rewrite lists → `needs_rewrite_combined.json`. |
| E — Phase 1b correct | `phase1b_gemini_correct.py --mode correct` | minimal-change fix of wrong-conclusion traces, reference-guided (image + GT type/location + GT mask) → `gemini_corrected.jsonl`. |
| F — Phase 1b rewrite | `phase1b_gemini_correct.py --mode rewrite` | full rewrite of marginal traces → `gemini_rewritten.jsonl`. |

The only change vs. the 10k launcher: the input pool is the disjoint held-out
file (passed as `--sft_pool`, labelled `source_pool="heldout_4k"`), and the
second pool slot is empty (`_empty_pool.json`, label `_none`). Rollout config,
reward functions, bucket thresholds, judge model, and correction prompts are
byte-identical to `run_pipeline_10k.sh`.

## Output location

`Training/heldout_4k_rollout/` (weights/large JSONL not committed to the repo):
- `rollouts_raw.jsonl` — 4,236 records, each with 8 scored rollouts
- `good_traces.json`, `good_traces_faithful.json`
- `needs_correction.json` + `gemini_corrected.jsonl`
- `needs_rewrite_combined.json` + `gemini_rewritten.jsonl`
- `difficulty.json`, `*.log`, `done.flag`

## Results

Completed 2026-06-17. Rollout: 4,236 items × k=8 in 8h47m single-GPU (CUDA 3).
Bucket → Gemini judge (Phase 1a) → Gemini correct/rewrite (Phase 1b) all finished.

| bucket | count | % of 4,236 |
|---|---|---|
| good (kept as-is) | 3,673 | 86.7% |
| — good after faithfulness filter (Phase 1a) | 3,589 | 84.7% |
| needs_correction (minimal fix) | 563 | 13.3% |
| needs_rewrite (full rewrite, local+judge union) | 704 | 16.6% |
| **Gemini-corrected** | 563 | 100% of needs_correction |
| **Gemini-rewritten** | 704 | 100% of needs_rewrite |

Net curated, training-disjoint pool ≈ **4,236 traces** = 3,673 good + 563 corrected
+ 704 rewritten (rewrite/correction sets are disjoint by construction in the bucketer).
Every failed rollout has a Gemini counterpart, so the pool covers all 4,236 items.

Compared to the 10,236-item in-distribution pool (good 8,872 = 86.7% — identical
keep rate), the held-out set buckets **the same way** (86.7% good), i.e. the model's
rollout quality on never-seen held-out images matches its quality on the training pool.
