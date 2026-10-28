# MMAD leakage side-experiment (11 Sep 2026)

Question: how much does a model gain on the MMAD benchmark when a small part of MMAD itself is in
its training set, the way OmniAD (v1, "one example per category from MMAD" in both SFT and GRPO)
and AnomalyR1 (600 images from the four MMAD source datasets) did?

## Setup

- Traces: one six-phase trace per MMAD image (8,293 of 8,366, the rest lack a verdict question
  or a mask), written by gemini-3.6-flash at thinking level MINIMAL with the thesis recipe:
  inspector_prompt_test_v2 system prompt, the v4 user prompt, batches of 10, three images per
  anomalous sample (original, red mask overlay, normal reference from MMAD's similar_templates),
  plus the MMAD answers (defect type, location, appearance, effect) as INTERNAL hints.
  Mean think length 113 words (anomalous) and 127 words (normal). 8 shards, 61 min, 55 structure
  rejects retried, none skipped. Generator: generate_mmad_traces_v4.py, traces in traces/.
- Run 1 (unbalanced): 1,600 training images, stratified by dataset, product and label at 19.3 %
  of every stratum, so the label ratio is MMAD's own (61 % anomalous overall, 74 % on DS-MVTec).
  Test set: the other 6,693 images. split.json.
- Run 2 (balanced, 6 epochs): 800 anomalous + 800 normal over the same products, at most 60 % of
  a product's images of either label, 1,428 images shared with run 1's training set, 6,521
  held-out images shared with run 1. split_bal.json. 6 epochs, 300 steps, 2 h 31 min, train loss 0.782.
- SFT recipe identical to the thesis 6K frozen run (sft_abc_C.yaml): Qwen2.5-VL-7B-Instruct from
  base, frozen ViT, full fine-tune of projector and LM, Z3 CPU offload, lr 1e-5 cosine, effective
  batch 32 on two A6000, 262,144 pixel cap. Run 1: 4 epochs, 200 steps, 1 h 41 min, train loss 1.133.
- Evaluation: vLLM, thesis harness protocol (same processor caps, greedy, 1,024 tokens, system turn
  "Please answer by yes or no", training prompt), strict balanced accuracy (unparsed = wrong),
  on the held-out keys of each subset. eval_heldout_vllm.py, score_heldout.py.
  Checkpoints need a *_vllmcompat twin (transformers 5.0 config is unreadable by the 4.57 vLLM env).

## Run 1: held-out strict BA (unparsed = 0 everywhere)

| Subset (held-out n) | Base | Ep 1 | Ep 2 | Ep 3 | Ep 4 |
|---|---|---|---|---|---|
| DS-MVTec (1,346) | 69.68 | 67.91 | 72.15 | **76.56** | 74.94 |
| VisA (1,730) | 53.87 | 63.46 | 65.60 | 66.59 | **67.72** |
| GoodsAD (2,341) | 50.99 | 56.77 | 57.52 | 59.52 | **59.57** |
| MVTec-LOCO (1,259) | 50.44 | lost | 53.56 | **53.98** | 52.12 |

Epoch 1 LOCO was not scored: the trainer's save_total_limit rotation counted the compat twins as
checkpoints and deleted checkpoint-50 before that evaluation. Fixed for run 2.

Thesis reference points, full subsets, no MMAD image in training: base 69.08 / 53.79, SFT-6K
(Real-IAD) 80.16 / 64.78, KCR 82.80 / 72.07 on DS-MVTec / VisA.

## Run 1: operating point (held-out)

| Model | Subset | BA | TPR | TNR |
|---|---|---|---|---|
| base | DS-MVTec | 69.68 | 39.9 | 99.4 |
| ep 3 | DS-MVTec | 76.56 | 80.1 | 73.0 |
| ep 4 | DS-MVTec | 74.94 | 85.8 | 64.1 |
| base | VisA | 53.87 | 7.7 | 100.0 |
| ep 4 | VisA | 67.72 | 52.5 | 82.9 |
| ep 4 | GoodsAD | 59.57 | 48.7 | 70.5 |
| ep 4 | MVTec-LOCO | 52.12 | 68.0 | 36.3 |

The 74 % anomalous training prior on DS-MVTec shows as over-calling (yes-rate 66 to 73 %,
specificity down to 64). On VisA and GoodsAD (55 % anomalous in training) the model under-calls.
Run 2 removes the prior.

## Run 1: memorisation check (epoch 3 scored on its own training images)

| Subset | Train images | BA on train images | BA held-out | Gap |
|---|---|---|---|---|
| DS-MVTec | 324 | 80.17 | 76.56 | +3.6 |
| VisA | 411 | 66.52 | 66.59 | 0.0 |
| GoodsAD | 559 | 63.17 | 59.52 | +3.7 |
| MVTec-LOCO | 302 | 55.64 | 53.98 | +1.7 |

The gain is not memorisation of the seen images. It is familiarity with the benchmark's products,
camera set-ups and defect vocabulary, which is the leakage that "one example per category from
MMAD" produces.

## Run 2 (balanced 800/800, 6 epochs): held-out strict BA

| Subset (held-out n) | Base | Ep 1 | Ep 2 | Ep 3 | Ep 4 | Ep 5 | Ep 6 |
|---|---|---|---|---|---|---|---|
| DS-MVTec (1,367) | 69.74 | 70.47 | **79.16** | 75.78 | 75.23 | 78.00 | 77.55 |
| VisA (1,717) | 53.78 | 61.46 | 61.24 | 61.48 | **67.22** | 66.44 | 66.01 |
| GoodsAD (2,329) | 51.02 | 57.35 | 56.07 | 59.50 | 60.62 | **61.07** | 60.62 |
| MVTec-LOCO (1,264) | 50.48 | 53.07 | 54.29 | 52.74 | **54.70** | 52.48 | 53.17 |

TPR / TNR per epoch (base first):

| Subset | Base | Ep 1 | Ep 2 | Ep 3 | Ep 4 | Ep 5 | Ep 6 |
|---|---|---|---|---|---|---|---|
| DS-MVTec | 40/99 | 67/74 | 76/82 | 79/73 | 79/72 | 83/73 | 84/72 |
| VisA | 8/100 | 35/88 | 34/88 | 34/89 | 49/85 | 58/75 | 52/80 |
| GoodsAD | 3/99 | 28/87 | 33/80 | 27/92 | 37/84 | 43/79 | 44/78 |
| MVTec-LOCO | 1/100 | 11/95 | 43/65 | 49/57 | 44/66 | 53/52 | 53/53 |

With the 50/50 prior the DS-MVTec operating point at epoch 2 is 76 sensitivity / 82 specificity,
against 80 / 73 for the best unbalanced epoch. Balanced accuracy 79.16 on unseen DS-MVTec images
after 1,600 benchmark images is one point below the thesis Real-IAD SFT on 6,000 traces (80.16).

## Run 2: memorisation check (epoch 2 scored on its own training images)

| Subset | Train images | BA on train images | BA held-out | Gap |
|---|---|---|---|---|
| DS-MVTec | 303 | 80.55 | 79.16 | +1.4 |
| VisA | 424 | 62.66 | 61.24 | +1.4 |
| GoodsAD | 571 | 57.72 | 56.07 | +1.7 |
| MVTec-LOCO | 297 | 55.26 | 54.29 | +1.0 |

## The thesis KCR model on the same held-out keys (no MMAD image in its training)

Qwen2.5-VL-7B KCR checkpoint-376 (thesis headline 82.80 / 72.07), scored on the run-2 held-out keys of all four
subsets with the same harness. GoodsAD and MVTec-LOCO were never evaluated in the thesis.

| Subset | BA | TPR / TNR |
|---|---|---|
| DS-MVTec | 82.35 | 80 / 85 |
| VisA | 73.36 | 60 / 87 |
| GoodsAD | 58.43 | 43 / 74 |
| MVTec-LOCO | 50.18 | 27 / 74 |

Pooled over the four subsets (MMAD's Anomaly Discrimination definition, zero-shot): 65.96.
For comparison, pooled on the same keys: base 56.1, run 2 best epoch 65.4. OmniAD-7B reports 61.4 zero-shot and
68.8 one-shot on that column. A model that never saw MMAD is ahead of the benchmark-trained runs on DS-MVTec
and VisA, the two subsets closest to Real-IAD's product families, and behind on GoodsAD, the retail packaging.

## Both runs on the 6,521 held-out images they share

| Subset | Base | Run 1 ep 2 | ep 3 | ep 4 | Run 2 ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 |
|---|---|---|---|---|---|---|---|---|---|---|
| DS-MVTec | 69.66 | 72.28 | 76.75 | 74.79 | 70.36 | 79.11 | 75.79 | 75.09 | 77.97 | 77.42 |
| VisA | 53.87 | 65.59 | 66.68 | 67.98 | 61.35 | 61.15 | 61.55 | 66.95 | 66.44 | 65.94 |
| GoodsAD | 50.97 | 57.52 | 59.48 | 59.46 | 57.25 | 56.12 | 59.19 | 60.43 | 60.85 | 60.49 |
| MVTec-LOCO | 50.44 | 53.92 | 54.55 | 52.07 | 53.12 | 54.14 | 52.52 | 54.57 | 52.23 | 53.24 |

Balancing helps DS-MVTec (+2.4 at the best epoch) and GoodsAD (+1.4), costs nothing on LOCO, and
VisA ends level (67.98 vs 66.95) after a slower start. Epoch-to-epoch swings of 3 to 4 points on
DS-MVTec are single-seed noise at n = 1,367 and should be read as such.

## Reading

- 1,600 benchmark images, 19 % of MMAD, lift the base model by +6.9 (DS-MVTec), +13.9 (VisA),
  +8.6 (GoodsAD) and +3.5 (LOCO) points on unseen images of the same products.
- Against the thesis Real-IAD-only SFT on 6,000 traces, the MMAD-trained models are ahead on VisA
  (67.7 and 67.2 vs 64.8) and within one to four points on DS-MVTec (79.2 balanced, 76.6 unbalanced,
  vs 80.2) with a quarter of the data.
- OmniAD's SFT set is also 1.6K images, of which 38 to 146 came from MMAD (one per product, or one
  per product and defect type, the paper does not define "category"). This run is the upper end of
  that practice: same training-set size, all of it from the benchmark.

## Files

- traces/ (8,293 json), split.json, split_bal.json, mmad_sft_train1600*.json
- evals/{base,checkpoint-*}/eval_<subset>_heldout.json, evals/checkpoint-150/eval_<subset>_trainkeys.json
- evals_bal/{base,checkpoint-50..300}/eval_<subset>_heldout.json (run 2), compat/ (vLLM twins)
- outputs/sft_qwen25vl_7b_mmad_train1600 (run 1, checkpoints 100, 150, 200) and
  outputs/sft_qwen25vl_7b_mmad_train1600_bal_6ep (run 2, checkpoints 50 to 300)
- logs/ (generation, training, evaluation)
