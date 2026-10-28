---
license: apache-2.0
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
pretty_name: AnomalyThink
configs:
- config_name: anomalythink_full_14k
  default: true
  data_files: anomalythink_15k/combined_sft_train.json
- config_name: thesis_sft_6k
  data_files: anomalythink_6k/combined_6k_train.json
- config_name: thesis_grpo_4k
  data_files: anomalythink_15k/c1_only_fixed/grpo_train.json
- config_name: thesis_heldout_4k
  data_files: anomalythink_15k/c1_only_fixed/new_sft_c1_train.json
- config_name: kcr_qwen_6k
  data_files: iter2/sft_iter2_train.json
- config_name: teacher_ablation_arm_a
  data_files: teacher_ablation_abc/sft_A_kept_balanced.json
- config_name: teacher_ablation_arm_b
  data_files: teacher_ablation_abc/sft_B_kept_corrected.json
- config_name: kcr_llava
  data_files:
  - llava_kcr/sft_llava_A_kept.json
  - llava_kcr/sft_llava_B_kept_corrected.json
  - llava_kcr/sft_llava_C_train.json
- config_name: other_sharegpt_files
  data_files:
  - anomalythink_6k/sft_train.json
  - anomalythink_6k/iad_sft_6k_bareprompt_train.json
  - anomalythink_15k/sft_train.json
  - anomalythink_15k/c1_only_fixed/combined_sft_c1_train.json
  - iter2/heldout_4236_disjoint.json
  - iter2/iter2_v2_sft_train.json
  - iter2/sft_iter2_heldout_train.json
  - teacher_ablation_abc/sft_10k_max_polish_train.json
  - variety_star_6k/sft_train.json
  - variety_star_6k/variety_star_sft_6k.json
- config_name: other_grpo_pools
  data_files:
  - anomalythink_6k/grpo_train.json
  - anomalythink_15k/grpo_train.json
  - variety_star_6k/grpo_train.json
---

# AnomalyThink: reasoning traces for explainable industrial anomaly detection

**AnomalyThink** is a collection of structured reasoning traces for industrial anomaly detection (IAD), distilled from **Gemini 2.5-Flash** on **Real-IAD** images. Each example is a single-image inspection in which the assistant produces a `<think>` reasoning trace, a defect `<location>` and `<type>` (for anomalies), and a binary `<answer>` (defect / no defect). Research artefact from the MSc thesis *Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection* (TU Delft, 2026).

## Important: images are NOT included
Examples reference **Real-IAD** images by **relative path** (`Real-IAD/images/...`) in the `images` field, or by `image_id` and `image_path` in the GRPO-pool files. The image files themselves are **not** redistributed here. Obtain Real-IAD from its official source, place it so the relative paths resolve, and cite Real-IAD.

## What Keep-Correct-Revise (KCR) means

Several folders here are products of **Keep-Correct-Revise (KCR)**, the curation recipe of the thesis. An SFT plus GRPO policy generates its own reasoning traces on Real-IAD images. Each trace is then **kept** when the policy got the verdict right, **corrected** by a teacher model when the verdict is wrong, and **revised** by a teacher model when the verdict is right but the reasoning is only weakly grounded in the image. The **base** model is then fine-tuned from scratch on the curated result.

The arms A and B in `teacher_ablation_abc` are partial versions of the same loop, keep-only and keep-plus-correct. The full loop, with the revise step included, is what KCR means.

## Format
Two schemas are used, and the `configs` above keep them apart so the viewer can load each:

- **ShareGPT** (`messages`, `images`): LLaMA-Factory SFT format, a list of `{"messages": [...], "images": ["Real-IAD/images/..."]}`. The assistant message holds the trace: `<think>...</think>`, then for anomalies `<location>...</location><type>...</type>`, then `<answer>yes|no</answer>`.
- **GRPO pool** (`image_id`, `image_path`, `product`, `is_anomaly`, `question`, `answer`, `gt_label`): one row per prompt for the reinforcement-learning stage, with the reference trace in `answer`.

## The thesis splits and where they are
The thesis partitions the 14,472 traces into three disjoint, product-stratified, class-balanced splits. Use these files to reproduce it.

| thesis split | count (anomalous / normal) | file | config |
|---|---|---|---|
| full corpus, all three splits | 14,472 (7,247 / 7,225) | `anomalythink_15k/combined_sft_train.json` | `anomalythink_full_14k` |
| AnomalyThink-6K, SFT split | 6,000 (3,000 / 3,000) | `anomalythink_6k/combined_6k_train.json` | `thesis_sft_6k` |
| GRPO split | 4,236 (2,118 / 2,118) | `anomalythink_15k/c1_only_fixed/grpo_train.json` | `thesis_grpo_4k` |
| held-out split (SFT format, never used for training in the thesis) | 4,236 (2,129 / 2,107) | `anomalythink_15k/c1_only_fixed/new_sft_c1_train.json` | `thesis_heldout_4k` |
| KCR corpus, Qwen2.5-VL (Arm C, thesis headline) | 6,000 (3,000 / 3,000) | `iter2/sft_iter2_train.json` | `kcr_qwen_6k` |
| teacher ablation, Arm A (keep only) | 2,978 | `teacher_ablation_abc/sft_A_kept_balanced.json` | `teacher_ablation_arm_a` |
| teacher ablation, Arm B (keep + correct) | 4,369 | `teacher_ablation_abc/sft_B_kept_corrected.json` | `teacher_ablation_arm_b` |
| KCR corpus, LLaVA-OneVision (A, B, C) | 8,998 / 9,124 / 6,000 | `llava_kcr/*.json` | `kcr_llava` |

The "15K" model of the thesis (7B-frozen-15K) was trained on the full 14,472-trace file, that is the SFT split plus the GRPO and held-out splits folded in as SFT data. Note on `iter2/sft_iter2_train.json`: this is the corpus as trained on. A version with 316 `<location>` strings normalised to the nine grid names is in the GitHub repository (`traces/iter2/sft_iter2_train.json`), same images and verdicts.

## Other files (alternative assemblies, kept for reproducibility)
| files | contents |
|---|---|
| `anomalythink_6k/sft_train.json`, `anomalythink_6k/grpo_train.json` | the 6K images re-split 3,000 SFT + 3,000 GRPO, an earlier configuration, not the thesis split |
| `anomalythink_6k/iad_sft_6k_bareprompt_train.json` | the 6K SFT split with the bare question as prompt |
| `anomalythink_15k/sft_train.json`, `anomalythink_15k/grpo_train.json` | 10,236 SFT (6K plus held-out) and a 4,236 GRPO pool, an earlier configuration |
| `anomalythink_15k/c1_only_fixed/combined_sft_c1_train.json` | GRPO split in SFT format plus the KCR corpus, 10,236 rows |
| `iter2/heldout_4236_disjoint.json`, `iter2/sft_iter2_heldout_train.json`, `iter2/iter2_v2_sft_train.json` | files of the iterative SFT-RL experiment (thesis 6.5 and 6.7) |
| `teacher_ablation_abc/sft_10k_max_polish_train.json` | the full 10,236-image rollout pool after correction, unbalanced |
| `variety_star_6k/` | Real-IAD Variety STaR corpus (6,000) and its pools, future work in the thesis, not a thesis result |

## Models trained on this data

The same KCR recipe was run on two backbones. Both families are released.

### Qwen2.5-VL-7B backbone
| Model | DS-MVTec | VisA | Note |
|---|---:|---:|---|
| [`AnomalyThink-Qwen2.5-VL-7B-KCR`](https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-KCR) | 82.80 | 72.07 | **thesis headline**, SFT only |
| [`AnomalyThink-Qwen2.5-VL-7B-SFT`](https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-SFT) | 80.16 | 64.78 | plain SFT baseline |
| [`AnomalyThink-Qwen2.5-VL-7B-SFT-GRPO`](https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-SFT-GRPO) | 82.73 | 70.39 | the policy KCR rolls out from |
| [`AnomalyThink-Qwen2.5-VL-7B-KCR-GRPO`](https://huggingface.co/aacudad/AnomalyThink-Qwen2.5-VL-7B-KCR-GRPO) | 82.95 | 72.62 | research preview, not a thesis result |

Balanced accuracy on the MMAD DS-MVTec (1,670) and VisA (2,141) subsets, one common harness.

### LLaVA-OneVision-7B-SI backbone
Cross-architecture replication of the same recipe, on the backbone IAD-R1 uses. Traces in `llava_kcr`.

- [`AnomalyThink-LLaVA-OneVision-7B-KCR`](https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-KCR)
- [`AnomalyThink-LLaVA-OneVision-7B-SFT`](https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-SFT)
- [`AnomalyThink-LLaVA-OneVision-7B-SFT-GRPO`](https://huggingface.co/aacudad/AnomalyThink-LLaVA-OneVision-7B-SFT-GRPO)

Note on DS-MVTec numbers for this backbone: the LLaVA-OneVision pretraining mix contains MVTec-derived rows, so DS-MVTec results on any LLaVA-OneVision model carry a contamination caveat. VisA is unaffected.

## Citation
```bibtex
@mastersthesis{acudad2026anomalythink,
  title  = {Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection},
  author = {Acudad, Adnane},
  school = {Delft University of Technology},
  year   = {2026}
}
```

## License
Apache-2.0 for the reasoning traces. Real-IAD images are under their own licence and are not included.
