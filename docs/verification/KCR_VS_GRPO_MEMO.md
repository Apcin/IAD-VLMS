# What the thesis shows about KCR versus GRPO, and how to frame it (5 Sep 2026)

Read: abstract through conclusion from the live .tex, plus appendix M. Papers read by Gemini 3.8 Flash from page images: DeepSeekMath (2402.03300, pp. 1 to 24), Dr.GRPO (2503.20783), RAFT++ / Minimalist (2504.11343), ReST-EM (2312.06585), DeepSeek-R1 (2501.12948), Gao et al. over-optimisation (2210.10760), SFT-memorises-RL-generalises (2501.17161), RAFT (2304.06767). Quotes below are from those pages. Raw notes in `thesis_figures_v2/audit/P_papers.txt`.

## 1. The whole thesis, read end to end

The argument is one ladder, Base to SFT to SFT+GRPO to KCR, told the same way in the abstract, §1.4, Figure 1.1, §6.1, §6.14, §7.5 and §8.1. It is consistent in structure. It is not yet consistent in what it claims the last rung is, and that is the thing to fix before the defence.

Three places say slightly different things about KCR:
- Abstract: "The last and best stage is a supervised one. What GRPO gave us was not the final model but the data we used to train it."
- §6.8 closing: "What the reinforcement-learning stage bought, in our runs, was a way to reach a level that a better-constructed supervised corpus reaches directly."
- §7.5: "did GRPO ever genuinely add value, or was the published lift an artefact of a sub-optimal SFT baseline?"

The first is right and should be the frame everywhere. The second implies the corpus could have been built without GRPO, which the thesis never tested and the data argues against (Arm A, pure rejection sampling on the same rollouts, is 3.8 points behind KCR on DS-MVTec). The third asks a question the data cannot answer and invites the committee to answer it for you.

## 2. How to frame the corrected LLaVA finding

The clean numbers (strict, non-answer = wrong):

| LLaVA-OneVision-7B | DS-MVTec | VisA |
|---|---|---|
| SFT ep1 | 85.91 | 68.26 |
| SFT + GRPO | 87.66 | 72.38 |
| KCR first build (leaky), ep4 | 88.45 | 74.25 |
| KCR corrected, ep2 (DS-best rule) | 87.32 | 72.65 |
| KCR corrected, ep4 (VisA-best) | 86.60 | 74.29 |

What changed with the correction: on VisA nothing (74.29 vs 74.25). On DS-MVTec, KCR fell from +0.79 above SFT+GRPO to between −0.34 and −1.06 below it.

The framing that survives:

"On the second backbone the corrective corpus beats the reinforcement-learned policy on the clean benchmark, VisA, by 1.91 points at matched training length, and ties it on DS-MVTec, a benchmark that carries a contamination caveat for this backbone. The first build of this corpus, which drew 2,484 of its 6,000 images from the GRPO split, scored 1.85 points higher on DS-MVTec and the same on VisA. The DS-MVTec margin was bought by the extra images, not by the recipe."

Three rules follow for the text:
1. Every LLaVA claim goes on VisA. §6.1 already commits to this. The abstract, Figure 1.1 footer and §8.1 (6) currently make it on both. Change those three to VisA.
2. Use the corrected build in the §6.9 KCR row and move the first build to Appendix M as "the defective first build". The reverse, which is the current state, has the thesis citing numbers it knows come from a flawed corpus.
3. Under the chapter's own DS-best selection rule the corrected model is ckpt-376, 87.32 / 72.65. That still beats SFT+GRPO on VisA (+0.27) but the margin is inside noise. If the VisA-best rule is used the margin is +1.91. Say which rule and why. My recommendation: keep the DS-best rule for consistency with every other row, report 87.32 / 72.65 in the table, and state in the text that the VisA optimum is at epoch 4 at 74.29, as §6.1 already promises to do "where a configuration's VisA peak falls at a different epoch".

The type vocabulary finding fits in one sentence in §6.6 or §7.5: "The correction step also broadens the defect vocabulary: 45% of the teacher-written corrections name a type outside Real-IAD's eight classes (Stain, Chip, Crack, Broken), against 3% of the rewrites and 1% of the model's own kept rollouts." It is a nice mechanistic detail about what the teacher does, and it is harmless to the metric.

## 3. Does the thesis prove that carefully curated traces approximately match GRPO?

Not as stated. It proves something narrower and more defensible.

What the data supports:
- On Qwen, one KCR SFT run matches the SFT+GRPO checkpoint on DS-MVTec (82.80 vs 82.73) and beats it on VisA (72.07 vs 70.39).
- On LLaVA, the corrected KCR ties SFT+GRPO on DS-MVTec and beats it on VisA.
- Arm A, the same rollouts with keep-only filtering and no teacher, gets 79.01 / 68.83. That is 3.8 / 3.2 below KCR and 3.7 / 1.6 below SFT+GRPO.
- GRPO on top of KCR, 4 epochs, three estimators, never exceeds the init on DS-MVTec.

What "curated traces match GRPO" would need and does not have:
- A curated corpus that does not come from the GRPO policy. Every KCR item is a rollout of the SFT+GRPO checkpoint or a teacher edit of one. The thesis has no "curated traces without GRPO" arm. So the honest statement is that the GRPO policy's rollouts, once curated, are a better SFT corpus than the GRPO policy is a model. That is the abstract's sentence and it is exactly right.
- The comparison is at equal image set but not at equal compute. KCR costs one GRPO run (24.7 h) plus 81,888 rollouts plus the teacher calls plus a 5.1 h SFT. SFT+GRPO costs the GRPO run alone. KCR is the more expensive pipeline, not a cheaper substitute. Say so in §7.5.

The literature gives you the mechanism, and it is worth one paragraph in §7.5 because it turns an empirical curiosity into an expected result:
- RAFT++ (2504.11343, p. 6): rejection-sampling fine-tuning reaches 56.1 against GRPO's 56.3 on Qwen2.5-Math-7B. "RAFT, which trains only on positively rewarded samples, yields competitive performance than GRPO and PPO" (p. 1). GRPO's remaining edge comes from negative samples keeping entropy up (p. 8) and from "discarding prompts with entirely incorrect responses" (p. 1).
- DeepSeekMath (p. 20): Online RFT "is comparable to RFT in the early stage of training but gains an absolute advantage in the later stage", and GRPO surpasses Online RFT because it "uniquely adjusts its gradient coefficient based on the reward value", while RFT "does not penalize incorrect responses and uniformly reinforces all responses with correct answers". Also (p. 21): "RL enhances Maj@K but not Pass@K ... the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities."
- ReST-EM (p. 5): each improve step "fine-tunes the base model instead of the model obtained from the previous ReST iteration", for "much better transfer performance on held-out tasks". This is the §6.7 reset-to-base result, published two years earlier. Cite it there.
- DeepSeek-R1 (p. 6, Fig. 2): the third stage is rejection sampling from the RL policy, then SFT of the base model on the filtered data. KCR is that stage with an external teacher added.

So the thesis reading is: KCR is rejection-sampling fine-tuning (RAFT / ReST-EM / R1 stage 3) with a teacher that supplies what the negative samples supply in GRPO, namely signal on the items the policy gets wrong. Arm A shows the rejection-sampling part alone is not enough on this task. Arm B and C show the teacher closes the gap. That is a clean story, it is supported by four papers, and it does not require claiming that curation replaces RL in general.

Suggested sentence for §7.5 and §8.2 RQ2: "KCR is the rejection-sampling-and-retrain stage of the DeepSeek-R1 pipeline with a teacher in the loop. Its rollouts come from the GRPO policy, so GRPO is upstream of the best model rather than absent from it. What the experiments show is that on this task the policy's curated rollouts are a better training signal than a second pass of the policy-gradient objective on the same policy."

## 4. GRPO on top of KCR: it was run, and what remains is the β question

It was run. §6.8 reports GRPO from Arm-C ckpt-376 for 4 epochs (2,120 steps, ckpt-265 to 2120), plus three 120-step estimator probes. All at β = 0. Every checkpoint is below the init on DS-MVTec (−1.6 to −4.0). VisA is +0.61 at ckpt-265 and negative after.

Would a KL penalty have changed it? The logs say no, for three reasons:
- The drop appears at negligible KL. GRPO-on-C is at KL 0.05 per token at ckpt-265 where DS-MVTec is already −2.19. The 120-step probes drop within 20 to 40 steps at KL below 0.01. Production run 2, which helped, reached KL 0.19. A penalty that held the policy inside KL 0.01 would hold it at the init.
- The gradient is running out, not running away. Batch reward std on the strong init falls from 0.455 to 0.097 over the run while mean reward reaches 2.49 of 3.0. Format is saturated at 1.0 and the verdict is right on most of the pool, so most groups carry no advantage. What is left is the ambiguous tail, and that is where the policy moves. Dr.GRPO (p. 6) names this the "question-level difficulty bias": groups with near-zero std get the largest weight. Removing the std division (their fix, the Dr.GRPO probe in §6.8) did not help, which says the problem is the signal, not the normalisation.
- Gao et al. (p. 7): "The KL penalty only causes the gold RM score to converge earlier, but does not affect the KL-gold reward frontier, and so the effect of the penalty on the gold score is akin to early stopping." And (p. 2): larger policies "benefit less from optimization against an RM ... but lead to very similar amounts of overoptimization ... and KL distance at which the maximum gold RM score is attained." Dr.GRPO (p. 5) drops β for rule-based rewards because there is no reward-model distribution shift to guard against. DeepSeek-R1 uses β = 0.001 and resets the reference every 400 steps, which is the opposite of holding the policy still.

Cost if you want the datapoint anyway: Qwen GRPO runs at 83.85 s/step on GPUs 1 and 2. One epoch from Arm-C ckpt-376 with β = 0.04 (DeepSeekMath's value) is 530 steps, about 12.3 h, plus four checkpoint evals at about 70 min each. Start tonight, numbers Sunday morning, before the defence. It adds one line to Table 6.x and closes the question if a committee member asks "did you try a KL penalty". trl's GRPOConfig has `beta`, the launch script sets `beta=0`, and the reference model is already loaded because the KL is monitored, so no code change and no memory change.

My recommendation: run it, one epoch, β = 0.04, everything else identical, and write it up as one sentence in §6.8 whatever it shows. The expected outcome is a flatter version of the same curve. If instead it improves on the init, that is a real result and it belongs in the thesis. Either way §7.4's line "adding a small KL penalty (β ≈ 0.05 to 0.2) is an obvious safeguard worth investigating" stops being a promise.

What would more plausibly help than β, for the future-work paragraph: dynamic sampling that drops all-correct and all-wrong groups (Reinforce-Rej in RAFT++ p. 10, DAPO), a reward with headroom on the strong init (a graded 3x3 location IoU, §7.3 already proposes this), and a difficulty-aware prompt set. The thesis already names two of these in F2. Add the first.

## 5. Text changes this implies (not applied)

1. Abstract, Figure 1.1 footer, §8.1 (6): LLaVA claim on VisA only. Replace "88.45 / 74.25, above 87.66 / 72.38 on both" with the corrected numbers and "on VisA".
2. §6.9 table: KCR row becomes the corrected build. First build moves to Appendix M with the defect note. Three qualifications paragraph loses "the DS-MVTec column of the winning arm was still rising", which was a first-build observation.
3. §6.8 last paragraph and §7.5 second paragraph: replace "reaches directly" and "did GRPO ever genuinely add value" with the R1-stage-3 framing above.
4. §6.7: cite ReST-EM p. 5 for reset-to-base. It is already in the bib as singh2023rest.
5. §7.5: one paragraph on cost. KCR is the more expensive pipeline.
6. §7.4 and §8.4: after the β = 0.04 run, replace the "worth investigating" line with the result.
7. §6.6 or §7.5: the one-sentence type-vocabulary finding.
8. Background §2.4.3: add RAFT++ (Xiong et al. 2025) and the DeepSeekMath Online-RFT result as the published precedent for "rejection sampling approaches GRPO".

## 6. OpenVLThinker v1 and v2, read by Gemini (added on request)

Raw notes in `thesis_figures_v2/audit/P_openvlthinker.txt`.

### v1 (Deng et al. 2025, arXiv 2503.17352)
- Loop: distil CoTs from a text-only R1 model (QwQ-32B) given image captions, keep "the shortest reasoning chain that correctly arrives at the final answer" (p. 6), SFT 3k examples (p. 21), then GRPO with outcome verification. Later iterations sample traces from the previous iteration's RL model on 3,000 data points (p. 2, 7).
- Restart every time: "Iter(i) is always fine-tuned from the base model Qwen2.5-VL-7B, with its training data generated from Iter(i-1)" (Fig. 3, p. 4). "To maintain stability, we retrain the model from scratch at each iteration with the newly generated dataset" (fn. 1, p. 7).
- Numbers, MathVista, 7B captions (Table 10, p. 23): SFT-Iter1 63.4 to GRPO-Iter1 66.6, SFT-Iter2 67.5 to GRPO-Iter2 70.9, SFT-Iter3 69.5 to GRPO-Iter3 71.7. So RL adds +3.2, +3.4, +2.2 per iteration, and each new SFT-only lands above the previous RL model's SFT but below its RL. The SFT-only of iteration i+1 (67.5, 69.5) is roughly the RL of iteration i (66.6, 70.9). That is the same "curated rollouts of the RL policy make an SFT that reaches the RL level" pattern as this thesis, one iteration at a time.
- Roles: "SFT serves as an inductive prior that highlights these reasoning actions ... Without this SFT step, launching RL from scratch forces the model to search through a prohibitively large space" (p. 2). "RL primarily serves to further refine and enhance performance" (p. 6).
- Where RL did not help: "RL training with easy-level data results in ineffective performance gain" (p. 10), and hard-after-medium gave −0.3 (Fig. 9, p. 11). Nothing on KL, over-optimisation, or a strong-init failure.
- Hyperparameters: SFT LR 5e-7, 1 epoch, batch 32. GRPO LR 1e-6. Group size and beta not in the paper.

### v2 (Hu et al. 2026, arXiv 2604.08539)
- Not an iterative SFT-RL paper. RL only, one epoch, from Qwen3-VL-Instruct-8B, on a filtered subset of OneThinker-600k, AdamW, batch 128, LR 2e-6 (p. 7). No SFT stage of its own.
- G2RPO as this thesis implements it: rank to quantile with the 0.5 offset, tie averaging (p. 4 to 5). Worked example (Fig. 2, p. 2): rewards [0,0,0,0,1] give vanilla [−0.5 x4, 2.0] and G2RPO [−0.32 x4, 1.28].
- KL: "we disable KL regularization and apply dynamic data filtering, actively discarding rollouts that are uniformly correct or incorrect to maintain high-quality gradient signals" (p. 7). That is DAPO dynamic sampling, exactly the fix suggested in Section 4 above.
- Rewards: accuracy, a length reward (Eq. 9, p. 6), format, and a structure reward for grounding (p. 16 to 17). Plus task-level entropy shaping.
- Gains over the instruct baseline (Table 4, p. 9): general VQA 71.3 to 77.9, math 59.2 to 66.2, chart 69.9 to 76.0, grounding 87.1 to 90.7, doc 86.8 to 91.4, spatial 60.9 to 63.6. "More modest gains observed in saturated or out-of-distribution domains" (p. 8).

### What this changes in the thesis
1. The thesis says (§2.4.2, F1) that OpenVLThinker "showed that iterative SFT-RL cycles outperform single-round SFT+RL". Table 10 supports that (66.6 to 70.9 to 71.7 across three RL rounds). Keep it. But add the number that matters for this thesis: in OpenVLThinker each SFT-only iteration lands at the level of the previous RL model (67.5 vs 66.6, 69.5 vs 70.9). That is the same observation as KCR. The thesis's contribution is that on IAD, with a teacher correcting the failures, the SFT-only step overtakes the RL model rather than just reaching it, and a further RL round does not add.
2. v2 disables KL and uses dynamic sampling with a length reward, from an instruct model, one epoch. The thesis's GRPO-on-C probe used G2RPO with KL monitored at zero and no dynamic sampling. The estimator was ported but the sampling was not. §6.8 should say that: "OpenVLThinker v2 pairs G2RPO with dynamic sampling that discards uniformly correct or incorrect groups. Our probe ports the estimator and not the sampling, and on the saturated Arm-C init most groups are uniformly correct, so the missing half is the one that matters." That is also the right first thing to try if a β = 0.04 run comes back flat.
3. §6.5 attributes the iter-2 failure to "rollout-and-filter concentrating the self-generated dataset on easy, already-correct examples". OpenVLThinker's filter is the opposite, keep the shortest correct chain, on 3k items, and their RL uses medium-difficulty data by design (p. 10). Say that the thesis's iter-2 kept 4,026 correct traces with no difficulty selection, which is one plausible reason it did not reproduce their gain.
