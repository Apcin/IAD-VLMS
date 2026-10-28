# GRPO KL-Penalty Drift Test (0.5 epoch on Arm-C)

**Date:** 2026-06-17 · **GPUs:** CUDA 1,2 · **Init:** Arm-C SFT `checkpoint-376`
(82.80 DS-MVTec / 72.07 VisA) · **Script:** `scripts/03_rollout_star/run_grpo_kl_halfep.sh`

## Question

GRPO on top of the strong Arm-C SFT init (`grpo_qwen25vl_7b_abc_C_grpo`, β=0, 4 epochs)
did **not** improve over SFT — VisA drifted down over epochs. Does turning on a **KL
penalty** (β>0, anchoring to the SFT init) arrest that drift?

## Setup (identical to abc_C except the KL penalty)

- Same dataset (`grpo_train.json`, 4,236 prompts), same prompt, same G=4, same LR (1e-6).
- Reward = **accuracy + format** (dropped the inert `reasoning` term — its :5100 judge is
  down so it returned a flat ~0.5 that GRPO's per-group advantage normalisation cancels).
- Embed server :5200 (Nomic type-similarity) UP for `accuracy_reward`.
- **β = 0.1** (prior runs used β=0). `--max_steps 265` (= 0.5 epoch; 4,236/EBS 8 = 530/ep).
- 5 checkpoints saved (53/106/159/212/265) for a drift curve; each eval'd DS→cuda1, VisA→cuda2.

## Results

| step | KL=0.1 DS | KL=0.1 VisA | no-KL (abc_C) DS | no-KL VisA |
|---|---|---|---|---|
| init (Arm-C 376) | 82.80 | 72.07 | 82.80 | 72.07 |
| 53 | 80.97 | 69.92 | – | – |
| 106 | 80.76 | 69.14 | – | – |
| 159 | 81.75 | 70.49 | – | – |
| 212 | 81.87 | 70.24 | – | – |
| **265 (0.5ep)** | **81.90** | **70.01** | 80.61 | 72.68 |
| 530 (1ep) | – | – | 80.32 | 70.34 |
| 1060 (2ep) | – | – | 81.20 | 69.17 |
| 1855 | – | – | 78.81 | 67.58 |
| 2120 (4ep) | – | – | 80.38 | 67.86 |

## Findings

1. **KL=0.1 is stable over 0.5 epoch** — no within-run drift; DS rises 80.97→81.90,
   VisA holds ~70. So the literal "does it drift?" answer is **no**.
2. **Neither run beats the SFT init (82.80/72.07)** — GRPO on the strong Arm-C init
   yields no gain, KL or not. Consistent with over-optimization / near-data-ceiling.
3. **Caveat — the test is short.** The no-KL run's real damage is a long-horizon VisA
   decline (72.68 → 67.86 across epochs 1–4). At matched 0.5 epoch the two are comparable
   (KL 81.90/70.01 vs no-KL 80.61/72.68). So this 0.5-epoch run shows KL is *short-term
   stable* but is **too short to prove KL arrests the multi-epoch collapse** — that needs
   a 2–4 epoch KL run.

## Reward summary (from train.log, logging_steps=1, 265 steps)

`results/grpo_abc_C_kl0.1_halfep/reward_curve.png` (3 panels: reward+components, KL, eval BA):
- total reward 1.42 → ~2.0; accuracy 0.73 → 1.12; format 0.69 → 0.88 — reward IS optimised.
- **KL stayed tiny (max 0.0106)** — β=0.1 glued the policy to the init; the policy barely moved.
- eval BA dipped then recovered (DS 80.97→81.90, VisA ~70), tracking just under the init.

## Why the step-53 dip (82.80 → 80.97) is expected

1. **Train/eval prompt mismatch** — GRPO trains on the short "…Are there any defects in the
   query image?" prompt; we *eval* on "Analyze the provided image of the {product}…". The first
   gradient steps adapt to the training prompt, perturbing eval-prompt behaviour.
2. **RL nudges a strong SFT init off its optimum immediately.** The KL penalty bounds the
   excursion to ~1–2pp (vs a collapse) and it then recovers. ~1pp on DS = ~17/1670 images,
   so part is eval noise. Net: it converges back toward ~82, not above the 82.80 init.

## Recommended next step (NOT yet run — decided 2026-06-17 to hold)

A **1-epoch** run would give a clean contrast vs the no-KL run at 530 steps (80.32/70.34,
already declining). Two caveats recorded for whenever we run it:
- **No clean resume** — `save_only_model=true` saved no optimizer state, so a 1-epoch run must
  be a **fresh run from ckpt-376 with `--max_steps 530`** (≈13h). This is also the correct way
  (proper LR decay over 530 steps).
- **β=0.1 will likely just recover to ~82.8 and plateau, not exceed it** — the KL≈0.01 shows the
  policy is *over-anchored*. To chase a gain above the SFT init, **lower β to ~0.04** (looser
  anchor, more room to move, higher drift risk). Launchers ready: `run_grpo_kl_halfep.sh`
  (edit `BETA=` and `--max_steps`), `watch_and_eval_grpo_kl.sh`.

## Artifacts (not committed; weights excluded)

`outputs/grpo_abc_C_kl0.1_halfep/`: `train.log`, `ba_table.txt`, `checkpoint-{53,106,159,212,265}/`
(+ per-checkpoint `eval_{dsmvtec,visa}_full_trainprompt.json`), `done.flag`.
