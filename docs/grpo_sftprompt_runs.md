# GRPO with the SFT/eval-aligned prompt — β-sweep + 3-epoch run

> **STATUS = FUTURE WORK, not a thesis claim** (decided 2026-06-20, see
> `../THESIS_HEADLINE_DECISION.md`). The thesis headline is **Arm-C SFT alone (82.80/72.07)**;
> the eval-aligned GRPO result here (which *edges* that ceiling) is preliminary/in-progress and
> is to be reported only once finalized + replicated. Do not write it as a result in the thesis body.

Experiments testing whether aligning the GRPO **training** prompt to the **eval**
prompt lets GRPO beat the Arm-C SFT ceiling (82.80 DS-MVTec / 72.07 VisA).
All from Arm-C `checkpoint-376`, dataset `grpo_train.json` (4,236, Gemini-distilled
gold traces), reward = accuracy + format (unweighted), full CPU offload.

## Background: the prompt-mismatch finding

The eval (`evaluate_qwen25vl_7b_trainprompt.py`, default mode) scores with:
```
system: "Please answer by yes or no"
user:   <image> "\nAnalyze the provided image of the {product}. Determine if there
         are any anomalies present. If an anomaly is detected, specify its type and
         location, and provide a detailed reasoning for your conclusion."
```
The 82.80 baseline was measured with this exact prompt (verified from the
`checkpoint-376` eval log: `no_system_prompt=False, grpo_eval=False`), so the
baseline comparison is **fair**. BUT the first GRPO runs (`--prompt_style sft`)
trained on the **user text only, no system message** → a real train/eval mismatch
(`grpo_ad.py` `prompt_style="sft"`). The new run fixes this with
`prompt_style="sft_sys"`, which is byte-identical to the eval prompt.

## Run 1+2 — β-sweep, 1 epoch, prompt_style=sft (NO system msg — mismatched)

`outputs/grpo_sftprompt_kl0.1` and `outputs/grpo_sftprompt_kl0.04`
(intermediate checkpoints pruned; only `checkpoint-530` + all eval JSONs kept).

| ckpt | β=0.1 DS | β=0.1 VisA | β=0.04 DS | β=0.04 VisA |
|---|---|---|---|---|
| 106 | 79.64 | 68.86 | 81.15 | 69.58 |
| 212 | 80.74 | 69.56 | 81.48 | 70.17 |
| 318 | 80.54 | 70.91 | 81.28 | 71.73 |
| 424 | 81.29 | 71.05 | 80.46 | 71.91 |
| 530 | **81.72** | 71.05 | **80.30** | 71.72 |

Reward/KL: `results/grpo_sftprompt_betacompare/reward_kl_curve.png`.
- β=0.1 (strict): reward 1.52→1.81, KL ~0.02, DS rises monotonically to 81.72, stable.
- β=0.04 (loose): reward 1.52→**1.91** (higher), KL ~0.025 (more drift), but DS
  **peaks at 81.48 then drifts down to 80.30** while VisA edges up — classic
  **reward-vs-eval decoupling / over-optimization** (more reward ≠ better model).
- **Neither beats 82.80/72.07.** Caveat: these trained *without* the eval's system
  message, so they are a *fair comparison* but not a *fully prompt-aligned* test.

## Run 3 — β=0.1, **3 epochs**, prompt_style=sft_sys (EVAL-ALIGNED) — IN PROGRESS

`outputs/grpo_sftprompt_kl0.1_sys_3ep`. Launcher
`scripts/03_rollout_star/run_grpo_sftprompt_3ep.sh`.
- **Train == eval** now (`sft_sys` = system "Please answer by yes or no" + Analyze,
  byte-identical to the eval). Trains on **CUDA 1,2**; 3 epochs (1,590 steps),
  checkpoint every 106 (15 checkpoints).
- **Decoupled eval watchdog on CUDA 3** (`scripts/03_rollout_star/watch_and_eval_cuda3.sh`):
  evals each new checkpoint as it appears — DS then VisA **sequentially** on GPU 3
  (never concurrent), so it coexists with other users' jobs on that card.
- Two questions: (a) does true alignment remove the early ckpt-106 dip seen with
  the mismatched prompt? (b) does β=0.1 over 3 epochs finally pass 82.8?
- **Expectation:** likely plateaus ~82 (KL-anchored to the init, tiny KL, OOD eval),
  so the value is the *KL-stability-over-3-epochs* curve more than a ceiling break.
- ETA ~36 h training (~81 s/step), eval keeps up in real time.

### Gotcha fixed during launch
First launch crashed with `ArrowInvalid: cannot mix list and non-list` — the
`sft_sys` system message had `content` as a bare string while the user message's
`content` is a list, which `datasets.map`/pyarrow can't serialize. Fix: system
`content` is a **list** of one text part (`[{"type":"text","text":"..."}]`) —
renders identically through the chat template. See `grpo_ad.py` `prompt_style=="sft_sys"`.

## See also
- `docs/grpo_beta_kl_explainer.md` — how the β KL-penalty enters the loss/gradient.
- `docs/grpo_kl_drift_test.md` — the earlier 0.5-epoch KL drift test.
