# Trace grounding audit

Does a trace in the AnomalyThink corpus actually describe the image it was written for?
This folder is the toolchain that asks that question over the whole 10,236-trace rollout
pool (6,000 SFT split + 4,236 GRPO split). It is an audit of the corpus produced by
stage 0, which is why it lives here.

**No audit results are committed.** The sweep was still running when this code was
added, and every rate the partial run produced is computed on far too few items to
quote. Run `run_full_sweep.sh` yourself before believing any number, and read the three
limits at the top of `aggregate_report.py` before writing a rate into a document.

## Pipeline

```
build_corpus.py        -> corpus.jsonl        resolve every trace to its image, mask,
                                              reference normal, gold label / type / location
format_check.py        -> format_check.jsonl  deterministic tag + parser check, no model.
                                              Regexes copied verbatim from
                                              scripts/02_grpo/stage_rl/reward.py
make_calib_sample.py   -> calib_200.jsonl     200-item stratified calibration sample
                                              (100 normal / 100 anomalous, all 30 products)
audit_traces_gemini.py -> *.jsonl             the Gemini grounding judge. Sharded,
                                              resumable, one fsynced line per item
run_full_sweep.sh                             the driver: prompt v2, 4 shards in sequence,
                                              3 passes (pass 2 and 3 redo errored records)
aggregate_report.py    -> audit_report.md     coverage first, then the rates
countable_check.py     -> countable_check.jsonl  model-free count check (transistor1 has a
                                              verified 9 leads per side / 18 total)
```

Calibration analysis: `analyse_calib.py` (failure families), `selfconsistency.py`
(pass 1 vs pass 2 on the same 40 items), `compare_v1_v2.py` (prompt v1 vs v2),
`extrapolate.py` (wall clock and token volume for the full pool).

Judge prompts: `prompt_grounding.txt` (v1) and `prompt_grounding_v2.txt` (v2, the one the
sweep uses).

## What you need that is not in this repository

- The two trace splits and the **raw Real-IAD images and masks**. `build_corpus.py` opens
  every image and mask, and we do not redistribute the pixels. See README section 3 of the
  repository root for the download and the path remapper.
- The intermediate `corpus.jsonl` (25 MB) and the calibration JSONLs. All regenerable from
  the scripts above, so none of them are committed.

## Credentials

`audit_traces_gemini.py` reads `GEMINI_API_KEY` from the environment and never writes it to
disk. `run_full_sweep.sh` will also source a mode-600 shell fragment if you point
`GEMINI_KEYFILE` at one (default `$HOME/.gemini_audit_key`). Nothing key-shaped belongs
inside this directory.

## Known limits of the method

These come from the 200-item calibration and they apply to every number this toolchain
produces. `aggregate_report.py` repeats them in its own header.

1. Count errors are a **lower bound**. The judge catches about 71 % of them and its own
   lead count on transistor1 is right only 55.7 % of the time. Publish a count-error rate
   from `countable_check.py`, never from the judge.
2. A single item's verdict flips about **7.5 %** of the time between two identical
   temperature-0 runs. Aggregate rates over 10,236 items are stable, one quoted verdict is
   not.
3. The `correct` category fired **0 times** in 200 calibration items, so the four-valued
   scale is three-valued in practice.
