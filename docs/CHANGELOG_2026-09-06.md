# Changelog, 2026-09-06: verification pass and repository sync

Everything added to the thesis between 3 and 6 September was verified against this workspace and brought into the repository. The verification report is `docs/verification/NUMBER_VERIFICATION_REPORT.md`. Two companion notes came along: `docs/verification/KCR_VS_GRPO_MEMO.md` (how to read KCR against GRPO, with the OpenVLThinker and RAFT comparison) and `docs/verification/WHOLE_THESIS_AUDIT.md` (the earlier whole-thesis audit).

## Results added (`results/`)

| folder | what | thesis use |
|---|---|---|
| `sft_llava_ov_7b_frozen_iad_sft_llava_iter1_C_original/` | LLaVA KCR, corrected corpus, 4 epochs, vLLM and HF files | Tables 6.9, 6.12, 6.13, Appendix M, Figures 1.1, 6.1 to 6.8 |
| `sft_llava_ov_7b_frozen_iad_sft_iter2/` | LLaVA on the Qwen KCR corpus, epoch 1 | Table 6.9 |
| `grpo_llava_ov_from_ep1_ep2/` | LLaVA SFT+GRPO restart | §6.9 |
| `grpo_llava_ov_from_ep1/`, `sft_llava_ov_7b_frozen_iad_sft_6k_train/` | HF-path and vLLM-path files added where missing | §6.9 two-paths paragraph |
| `gemini_25flash_eval/`, `gpt5mini_eval/` | proprietary zero-shot references | Table 6.14, Appendix G |
| `sft_filtered6kcc_from_base/` | retrain on verifier-passing traces | §7.2 |
| `grpo_probe_ctrl/`, `grpo_probe_drgrpo/`, `grpo_probe_g2rpo/` | 400-sample probe JSONs per 20 steps | Table 6.10, Figure 6.6 |
| `grpo_sftprompt_kl0.1_sys_3ep/` (all 15 checkpoints), `grpo_sftprompt_kl0.1/`, `grpo_abc_C_kl0.1_halfep/` | prompt-aligned and beta 0.1 GRPO-on-C runs | §6.8, §8.4 (inconclusive) |
| `explainability_multi/` | judge outputs for the LLaVA rows and the corrected KCR row, regenerated axis table | Table 6.13 |
| `thesis_figure_data/` | `figdata.json`, `loc_hit.json`, `type_sim.json`, the two generated tables | every Chapter 6 figure, Table 6.12, Appendix J.5 |
| `contamination_llava_ov_data/` | the 1,999-of-186,060 count and its code | §6.1 caveat |
| `sft_vs_kcr_pairs/` | indices of the 65 / 30 / 21 SFT-no, KCR-yes pairs | Figure 6.9, Appendix F.5 |
| `grpo_qwen25vl_7b_6k_frozenvision_run1/` | partial training log of the frozen-vision GRPO ablation | pending |

Each new folder has a `NOTE.md` with the numbers and caveats.

## Corpora added (`traces/`)

- `llava_kcr/`: the corrected LLaVA KCR corpus and the first (leaky) build.
- `grpo_split/grpo_train.json`: the file every GRPO run trained on. Not the same traces as `anomalythink_15k/grpo_train.json`.
- `rollout_pools/{qwen,llava}_phase0_10k/`: both 10,236 x 8 rollout pools, judge reports, kept / corrected / rewritten buckets (large files gzipped).
- `controls/`: verifier-filtered 6K, Balanced-192, labels-only.
- `iter2/sft_iter2_train.json` replaced by the location-spelling-fixed version, the previous file kept as `sft_iter2_train_v1_prepatch.json`.

## Scripts added (`scripts/`)

- `05_figures/thesis_figures_v2/`: the complete Chapter 6 figure pipeline (`registry.py`, `export_data.py`, `f1..f6`, `g1..g5`, `thesisify.py`, `loc_hit_table.py`, `type_sim_table.py`, `stages/f7_stages.py`, final SVGs). `05_figures/analyze_anomaly_types.py` (t-SNE of type strings).
- `03_rollout_star/build_llava_arms_original.py`, `armc_original_autostart.sh`: the corrected LLaVA corpus builder and its launcher.
- `04_eval/hf_llava_backfill.sh`, `evaluate_vllm_qwen3vl.py`, `run_on_folder.py`, `explainability_judge_multi_corrected.py`, `watch_and_eval_grpo_frozenvision.sh`.
- `02_grpo/run_grpo_7b_frozen_vision.sh` and the env-guarded `FREEZE_VISION_TOWER` block in `stage_rl/trainer/sc_grpo_trainer.py` (off by default).

Python copies take the workspace root from `WORK_DIR` (default the parent of this repository); shell launchers carry a header noting the same.

## Ledgers

- `NUMBER_PROVENANCE.md`: 2026-09-06 entry with every new number and file, figure-provenance rows, and a revised authority note (the thesis is authoritative where the older entries disagree).
- `CLAIMS_EVIDENCE.md`: Part 0b with the verification-pass claims and their status, inventory rows corrected (GRPO file, trace length).
- `UNVERIFIED.md`: section 5, what still cannot be backed from disk.
- `THESIS_HEADLINE_DECISION.md`: prompt-aligned run marked complete and inconclusive.
- `README.md`: section 0 pointer, corrected LLaVA rows, contamination count, layout table, corpora table.

## Thesis changes of the same day (local commit c7a4093 of the thesis repository, not yet pushed)

Strict scoring in Tables 6.6, 6.7 and 6.11; the prompt-aligned run noted in §6.8 and §8.4 as inconclusive; Arm A balance stated by verdict; the 15K under-training sentence replaced by what the logs show; pcb3 delta 7.5; reward-code line references; HF-vs-vLLM agreement range; 2.4x optimiser steps.

## Resolved later the same evening (thesis commit ff0c141)

Appendix C's base column moved to the Table 6.1 baseline file (69.08), the overall SFT gain reads +11.1 pp, and §6.11 / RQ4 report the Welch tests over all fifteen pairs per benchmark with the recomputed t values (15.6 / 6.5 against IAD-R1).

## Later the same evening (thesis commit 324fffe, repo 3308e54 and this commit)

Shared-set explainability judge: all nine rows scored on 90 + 47 identical images, paired differences replace the Welch and z tests, Table 6.13 rebuilt, own-100 table removed from the thesis, Figure 6.8 axis means updated, protocol-provenance paragraph added. Figure 4.1 generator fixed after a Gemini layout review (real rollout group, no overflow).

## Model names unified (thesis commits b116d51, 5c8e26f, a7f1fe7)

The thesis now uses four names for the Qwen2.5-VL-7B models, matching the figures: Base, SFT (ckpt-564), SFT+GRPO (ckpt-530) and KCR (ckpt-376), with a naming table in §6.1. "Run 2", "the production run" and "SFT-Iter1" are gone from the prose; "Arm C" survives only inside the teacher ablation. The folder names in `results/` are unchanged (`grpo_qwen25vl_7b_6k_frozen_ep3_full_run2` is SFT+GRPO, `sft_qwen25vl_7b_abc_C_full_patched` is KCR). Three appendix figures that carried the old labels were regenerated from the eval files: `scripts/05_figures/thesis_figures_v2/appendix/h1_grpo_curves.py` (Appendix H reward and KL curves) and `h2_iter2_overlays.py` (Appendix F overlays and Appendix D per-product VisA plot, SFT+GRPO against the post-GRPO refinement on the cleaned 6K pool). A Gemini pass over all 25 thesis figures confirmed no other figure carries an old name.

## 2026-09-07, early morning (thesis commits 8b00bcd, 41b4881, 5b18e0a, all on Overleaf)

Every example figure in the thesis redrawn in the house style: Figure 6.9, the 21 Appendix F.5 pairs, the Appendix F
galleries (now selected by judge score, one per product), the F.3 walk-through (cable poke_insulation/001 through
Base, SFT and SFT+GRPO), the F.4 overlays (lighter mask tint) and the five Appendix G teacher-against-student cases.
Generators: `scripts/05_figures/thesis_figures_v2/f8_pairs.py` and `f9_appendix_examples.py`; selections in
`results/sft_vs_kcr_pairs/`. Frozen-vision GRPO ablation, checkpoint-265 (epoch 1 of 4), strict: DS-MVTec 80.57,
VisA 67.25 (SFT start 80.16 / 64.78, SFT+GRPO ckpt-530 82.73 / 70.39). Files under
`results/grpo_qwen25vl_7b_6k_frozenvision_run1/` once the run completes.

## 2026-09-07 and 08 (thesis commits 8b00bcd through 3db679c, repo through this commit)

- Review pass on a ChatGPT review of the thesis: 33 items checked and applied (report in the thesis scratchpad, not shipped here). Substantive corrections shipped: the SFT+GRPO headline files were scored under the GRPO prompt (note in `results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/NOTE.md`, train-prompt VisA file added); the LLaVA base "50.00" row is a parser-fallback artefact; z-score advantages are bounded, not exploding; Listing 5.1 was an impossible decomposition; the t-SNE figure was drawn from an early 8,908-string pool (`results/anomaly_type_analysis/final_corpus/`).
- New results: LLaVA KCR on held-out Real-IAD (84.03), frozen-vision GRPO reference run (Appendix N, `results/grpo_qwen25vl_7b_6k_frozenvision_run1/`), Gemini 3.5 to 3.8 Flash zero-shot references (`results/gemini_flash_family_eval/`), type-reward similarity tables on the reward's own embedding path.
- Figures: Figure 6.9 and all appendix example figures redrawn in the house style (`scripts/05_figures/thesis_figures_v2/f8_pairs.py`, `f9_appendix_examples.py`), Figure 6.7 as a heat map, Figure H.1 regenerated (`appendix/h0_sft_curves.py`).
- Hugging Face dataset card rewritten with the thesis splits mapped to files and per-schema configs (`docs/HF_DATASET_CARD.md`).
- Repository history rewritten before going public: commit trailers removed, an internal handoff note and the `thesis_review/` drafts removed from every commit, the stray `add-interactive-viewers` branch merged into `main` as `docs/interactive/` and deleted. A full-history scan found no credential of any kind.
