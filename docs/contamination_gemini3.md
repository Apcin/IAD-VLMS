# Gemini-3-Flash-preview zero-shot on MMAD — and a benchmark-contamination flag

> **STATUS = REPO-ONLY, do NOT report in the thesis (decided 2026-06-20).**
> The thesis reports the two zero-shot proprietary references **Gemini-2.5-Flash (81.52/75.18)**
> and **GPT-5-mini (77.10/68.23)** in `tab:sota`. The Gemini-3 number below is held back because
> it is almost certainly inflated by pretraining contamination on the public MVTec-AD images
> (see analysis). Keep here as evidence / future-discussion material for Limitation L5.

## The numbers (all under the identical eval harness, same 1,670 DS / 2,141 VisA images)

| Model (zero-shot, structured prompt) | DS-MVTec BA | VisA BA | DS recall | DS specificity |
|---|---|---|---|---|
| GPT-5-mini (effort=low) | 77.10 | 68.23 | 96.1 | 58.1 |
| Gemini-2.5-Flash (thinking_budget=0) | 81.52 | 75.18 | 92.1 | **70.9** |
| **Arm-C SFT (our 7B student)** | **82.80** | **72.07** | — | — |
| **Gemini-3-Flash-preview (thinking_level=minimal)** | **93.20** | **80.29** | 93.4 | **93.0** |

Gemini-3: DS recall 93.4 / spec 93.0 / acc 93.2 (no_answer=1); VisA recall 84.2 / spec 76.4 / acc 80.7 (no_answer=1).

## Why this looks like contamination on DS-MVTec

1. **Magnitude.** Gemini-3 jumps **+11.7 pp over Gemini-2.5 on DS-MVTec** (and +10.4 over our
   fine-tuned student) to **93.2 % balanced accuracy — zero-shot, minimal thinking.** That is
   essentially at the ceiling of the MMAD binary-VQA format.
2. **Where the gain came from is the tell.** The DS gain is almost entirely **specificity**:
   70.9 -> 93.0 (**+22 pp**), while recall barely moves (92.1 -> 93.4). I.e. on *normal* MVTec
   parts, 2.5 false-alarmed 29 % of the time; 3 only 7 %. Near-perfect "this part is normal"
   discrimination on a **7-year-old public dataset**, from a barely-thinking model, is the
   fingerprint of having seen the images (and their normal/abnormal labels) during pretraining.
   A from-scratch reasoner does not get nominal-part calibration that good for free.

## Honest hedge (why it is *not* conclusive)

- **VisA only reaches 80.29** (not 93). If it had simply memorised "all the IAD benchmarks" both
  would be inflated; the large DS≫VisA gap is consistent with VisA being genuinely harder
  (multi-object, subtle), i.e. *some* of Gemini-3's strength is real capability.
- **MVTec-AD is a known-easy / near-saturated** benchmark, so a much stronger 2025/26 model
  *should* score high regardless.
- Memorisation vs. capability **cannot be separated** without a private / post-Gemini-cutoff
  benchmark. So the honest statement is: *likely meaningful MVTec exposure; partial-or-less on
  VisA; unprovable here.*

## Why we keep it out of the thesis

Dropping a 93.2 "teacher" row into `tab:sota` would read as absurd and invite exactly this
contamination objection. If used at all it belongs in the **L5 discussion** (`sec:disc-limitations`)
as a *cautionary* data point — "frontier models now post near-ceiling zero-shot scores on these
public benchmarks, which we attribute substantially to pretraining exposure, underscoring why
controlled Real-IAD→MMAD transfer (not raw benchmark score) is the meaningful measure" — **not** as
a result. Decision: hold until/unless we have a clean (private/post-cutoff) eval to deconfound it.

## Provenance / how to reproduce

- **Model:** `gemini-3-flash-preview` on Vertex AI, `thinking_level=minimal` (Gemini-3 control,
  *not* `thinking_budget=0`). The harness reads the Vertex project, location and service-account
  path from `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` and
  `GOOGLE_APPLICATION_CREDENTIALS`. Nothing is hardcoded, see `.env.example`.
- **Harness:** `scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py` with
  `--gemini --gemini-model gemini-3-flash-preview --gemini-thinking-level minimal`, plus the
  generic `--shard k --num-shards 4` sharding. Same structured
  `<think>/<type>/<location>/<answer>` prompt + GT + metrics as the Gemini-2.5 row, so directly
  comparable. Env: `llama_sft` (google-genai 1.72.0, transformers 5.0.0).
- **Run:** 8 shards (4 DS + 4 VisA), CPU-only. 603 retries on `429 RESOURCE_EXHAUSTED` (preview
  quota), all absorbed by backoff — `no_answer=1` per subset, so the BA is essentially clean.
- **Artifacts:** written to the workspace, not committed here:
  `$WORK_DIR/outputs/gemini3flash_eval/{ds,visa}_shard{0..3}of4.json` and the merged
  `eval_gemini3flash_minimal_{ds,visa}_merged.json` beside them.
- **Merge/BA:** `python scripts/04_eval/merge_api_eval.py <outdir> gemini3flash_minimal`.

## See also
- GPT-5-mini reference run: `$WORK_DIR/outputs/gpt5mini_eval/`, merged with the same
  `scripts/04_eval/merge_api_eval.py` (that outdir is its default).
- Thesis Limitation L5 (`chapters/07_discussion.tex`, `sec:disc-limitations`): trace-generator /
  evaluation contamination caveat — this finding is direct evidence for it.
