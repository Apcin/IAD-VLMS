---
license: cc-by-nc-sa-4.0
task_categories:
- image-text-to-text
- visual-question-answering
language:
- en
tags:
- industrial-anomaly-detection
- reasoning
- chain-of-thought
- vision-language-model
- benchmark-leakage
- mmad
pretty_name: AnomalyThink-MMAD
configs:
- config_name: all_8293
  default: true
  data_files: mmad_all_8293.json
- config_name: run1_unbalanced_train_1600
  data_files: splits/run1_unbalanced_train_1600.json
- config_name: run1_unbalanced_test_6693
  data_files: splits/run1_unbalanced_test_6693.json
- config_name: run2_balanced_train_1600
  data_files: splits/run2_balanced_train_1600.json
- config_name: run2_balanced_test_6693
  data_files: splits/run2_balanced_test_6693.json
- config_name: one_per_product_and_defect_144
  data_files: splits/one_per_product_and_defect_144.json
- config_name: one_per_product_39
  data_files: splits/one_per_product_39.json
- config_name: traces_with_provenance
  data_files: traces_8293.jsonl
---

# AnomalyThink-MMAD: reasoning traces for the MMAD benchmark

**{{DATASET_ID}}** holds one structured reasoning trace for 8,293 of the 8,366 images of the
**MMAD** benchmark (DS-MVTec, VisA, GoodsAD, MVTec-LOCO), written with the same teacher recipe as
the [AnomalyThink](https://huggingface.co/datasets/aacudad/AnomalyThink) corpus of the MSc thesis
*Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection* (TU Delft, 2026).
Each trace is a single-image inspection: a `<think>` block, and for anomalies a `<location>` on a 3x3 grid
and a `<type>`, then a yes/no `<answer>`. The MMAD annotations (mask, defect type, location, appearance,
effect) were given to the teacher, so the traces are grounded in the benchmark's own ground truth.

**Read this before training on it.** MMAD is an evaluation benchmark. A model fine-tuned on any part of
this corpus and then scored on MMAD has seen the benchmark's products, cameras and defect vocabulary,
and its MMAD numbers are not comparable with models that have not. We used the corpus to measure
exactly that effect (section *Leakage test* below). No thesis model was trained on any MMAD image,
and the checkpoints fine-tuned on this corpus are deliberately not released.

## Important: images are NOT included
Every example references an MMAD image by relative path (`MMAD/DS-MVTec/bottle/image/broken_large/000.png`),
which is the key of the image in MMAD's `mmad.json`. Obtain MMAD from its official source
(<https://huggingface.co/datasets/jiang-cc/MMAD>, CC BY-NC-SA 4.0) and place it so the paths
resolve. This dataset inherits MMAD's non-commercial licence.

## How the traces were written
- Teacher: `gemini-3.6-flash` (Vertex AI), thinking level MINIMAL, JSON output, batches of 10.
- System prompt: `inspector_prompt_test_v2` of the thesis (six-phase inspection, XML tags,
  hard rules, auto-reject rules), user prompt of the v4 Real-IAD generator.
- Inputs per anomalous image: the image, the same image with the MMAD ground-truth mask as a red
  overlay, and a normal reference image of the same product taken from MMAD's `similar_templates`.
  Normal images: the image only.
- Hints: the MMAD multiple-choice answers for defect type, location, appearance and effect were
  given to the teacher as internal labels and are stored in `hints` of `traces_8293.jsonl`.
- Output: `<think>...</think>` then, for anomalies, `<location>` on a 3x3 grid, `<type>` from the
  thesis defect vocabulary, and `<answer>Yes</answer>`, otherwise `<answer>No</answer>`.
  Mean think length 113 words (anomalous) and 127 words (normal).
- Excluded: 69 images without a verdict question in MMAD and 4 MVTec-LOCO anomalies without a mask.

## Files
| file | rows | content |
|---|---|---|
| `mmad_all_8293.json` | 8,293 | ShareGPT (LLaMA-Factory) records, all traces |
| `traces_8293.jsonl` | 8,293 | one record per image with `hints`, `dataset`, `product`, `is_anomaly`, teacher, and the split membership of both runs |
| `splits/run1_unbalanced_train_1600.json` / `_test_6693.json` | 1,600 / 6,693 | 19.3 % of every (dataset, product, label) stratum, MMAD's own label ratio |
| `splits/run2_balanced_train_1600.json` / `_test_6693.json` | 1,600 / 6,693 | 800 anomalous + 800 normal, at most 60 % of a product's images of either label, 6,521 test images shared with run 1 |
| `splits/one_per_product_and_defect_144.json` | 144 | one image per (dataset, product, defect folder), the OmniAD "one example per category" reading |
| `splits/one_per_product_39.json` | 39 | one image per product, the other reading |
| `splits/split_keys.json` | | the key lists of every split |
| `RESULTS.md` | | the experiment: recipe, epoch tables, sensitivity/specificity, memorisation checks |
| `system_prompt_inspector_v2.txt` | | the exact system prompt the teacher was given (the user prompt is built in the generator) |

## Leakage test: what training on part of MMAD buys
Several published VLM anomaly detectors put MMAD images into their training data (OmniAD v1: "one
example per category from MMAD" for both SFT and GRPO; AnomalyR1: 600 images from the four MMAD
source datasets) and report MMAD scores. To measure the effect we fine-tuned Qwen2.5-VL-7B with the
thesis SFT recipe on 1,600 of these traces (two splits: MMAD's own label ratio, and 800/800) and
scored the other 6,693 images of the same products with the thesis harness (strict balanced
accuracy, unparsed = wrong, zero-shot, single image). Full tables in `RESULTS.md`.

Strict balanced accuracy on unseen images of the same products:
| Subset | Qwen2.5-VL-7B base | 1,600 MMAD traces, balanced, best epoch | thesis SFT on 6,000 Real-IAD traces (no MMAD) |
|---|---|---|---|
| DS-MVTec | 69.7 | 79.2 | 80.2 |
| VisA | 53.8 | 67.2 | 64.8 |
| GoodsAD | 51.0 | 61.1 | n/a |
| MVTec-LOCO | 50.5 | 54.7 | n/a |

Scoring the same checkpoint on its own training images gives only 1 to 2 points more than on the
held-out images, so the gain is familiarity with the benchmark's products, not memorisation.
The thesis KCR model, which never saw an MMAD image, scores 82.4 / 73.4 / 58.4 / 50.2 on the same
held-out keys (DS-MVTec / VisA / GoodsAD / MVTec-LOCO); pooled over the four subsets that is 66.0
against 65.4 for the best benchmark-trained run and 61.4 zero-shot for OmniAD-7B.

## Reproduce the traces and the experiment
Code and the exact prompt are in the GitHub repository, folder `scripts/06_mmad`:
- generator: <https://github.com/aacudad/IAD-VLMs/blob/main/scripts/06_mmad/generate_mmad_traces_v4.py>
  (the user prompt, the three-image input and the hint lines are built inside `format_anomaly_prompt`)
- system prompt, the exact file the run used, also shipped here as `system_prompt_inspector_v2.txt`:
  <https://github.com/aacudad/IAD-VLMs/blob/main/scripts/06_mmad/inspector_prompt_test_v2_mmad_run.txt>
- split, SFT recipe, evaluation and scoring: `compile_split.py`, `compile_split_bal.py`, `sft_mmad_train1600*.yaml`,
  `eval_heldout_vllm.py`, `score_heldout.py` in the same folder, write-up in `RESULTS.md`
- per-checkpoint evaluation files: <https://github.com/aacudad/IAD-VLMs/tree/main/results/mmad>

```
export WORK_DIR=/path/to/your/clone   # MMAD under $WORK_DIR/reasoning_traces_gen/data/MMAD
python generate_mmad_traces_v4.py --shard_id 0 --total_shards 8   # one process per shard, Vertex AI credentials in GOOGLE_APPLICATION_CREDENTIALS
python compile_split.py && python compile_split_bal.py
```

## Citation
Cite the thesis and MMAD. Code: <https://github.com/aacudad/IAD-VLMs> (`scripts/06_mmad`).
