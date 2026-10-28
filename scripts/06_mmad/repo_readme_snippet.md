# MMAD reasoning traces and the leakage test (11 Sep 2026)

This folder generates one thesis-recipe reasoning trace per MMAD image (8,293 traces, published as
the AnomalyThink-MMAD dataset) and uses them for one question: how much does a Qwen2.5-VL-7B gain
on the MMAD benchmark when part of MMAD is in its training set? Published systems do this (OmniAD v1:
one MMAD example per category in SFT and GRPO; AnomalyR1: 600 images from the four MMAD source
datasets). It is a side experiment, not part of the thesis, no thesis model saw an MMAD image in
training, and the checkpoints fine-tuned here are deliberately not released.

Full write-up with all tables: [`RESULTS.md`](RESULTS.md). Dataset: <https://huggingface.co/datasets/aacudad/AnomalyThink-MMAD>.

## Pipeline

| step | script | notes |
|---|---|---|
| traces | `generate_mmad_traces_v4.py` + `inspector_prompt_test_v2_mmad_run.txt` (exact system prompt used) | thesis teacher recipe (v2 system prompt, v4 user prompt, image + red mask overlay + normal reference), MMAD answers as internal hints, `gemini-3.6-flash` MINIMAL, 8 shards |
| split | `compile_split.py`, `compile_split_bal.py` | 1,600 train / 6,693 test, stratified by dataset, product and label. Run 1 keeps MMAD's label ratio, run 2 is 800/800 |
| SFT | `sft_mmad_train1600*.yaml`, `run_sft_*.sh` | identical to the thesis 6K frozen recipe (`configs/sft/sft_abc_C.yaml`), 4 and 6 epochs |
| eval | `eval_heldout_vllm.py`, `eval_all_checkpoints.sh`, `eval_ckpts_on_gpu.sh` | thesis vLLM harness protocol, held-out keys of all four subsets. `MMAD_SPLIT` selects the split file |
| score | `score_heldout.py` | strict balanced accuracy on held-out keys, train keys (memorisation) and full subset |
| release | `build_release.py` | writes the HF staging folder and this repo folder, paths normalised to `MMAD/...` |

Set `WORK_DIR` to your clone of the cluster tree. Checkpoints saved by the transformers-5 training
env need a vLLM-compatible twin (stock 4.x config, symlinked weights), which the eval scripts build
in `compat/` outside the run directory (the trainer's `save_total_limit` counts any `checkpoint-*`).

## Data and results in this repo

- `traces/mmad/`: the 8,293 traces (`traces_8293.jsonl` with hints and provenance),
  ShareGPT files of every split, `split_keys.json`. Image paths are `MMAD/<mmad.json key>`.
- `results/mmad/`: per-checkpoint eval JSONs of both runs on the four held-out subsets,
  the train-key evaluations, the thesis KCR model on the same keys, and the two split files.

## Headline (strict BA on unseen images of the same products)

| Subset | base | run 2 balanced, best epoch | thesis SFT-6K (Real-IAD only) |
|---|---|---|---|
| DS-MVTec | 69.7 | 79.2 | 80.2 |
| VisA | 53.8 | 67.2 | 64.8 |
| GoodsAD | 51.0 | 61.1 | n/a |
| MVTec-LOCO | 50.5 | 54.7 | n/a |

Pooled over the four subsets (MMAD's Anomaly Discrimination definition), zero-shot: base 56.1,
run 2 best 65.4. OmniAD-7B reports 61.4 zero-shot and 68.8 one-shot on that column.
