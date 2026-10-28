# GRPO + the KL (β) penalty — how it works and where β acts

A self-contained explainer of the GRPO objective used in this thesis (TRL
`GRPOTrainer`, custom reward), focused on **how the β KL-penalty enters and
whether it affects the weight updates** (it does — directly). Math renders in
GitHub / VS Code markdown preview.

---

## 1. Setup — group sampling

For a single prompt $q$ (one image + the question), we sample a **group** of
$G$ completions from the rollout policy $\pi_{\theta_{\text{old}}}$:

$$\{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q), \qquad G = 4 \text{ here.}$$

Each completion is scored by the reward function (the accuracy + format sum,
max 3.0):

$$r_i = R(o_i; q).$$

GRPO's trick: instead of PPO's learned value network, it uses the **group mean
as the baseline**.

---

## 2. Group-relative advantage

Every token in completion $i$ gets the same scalar advantage — the group
z-score of its reward:

$$\boxed{\;\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon}\;}$$

**Consequence:** if all $G$ completions of a prompt get the *same* reward,
$\operatorname{std}\to 0$ so $\hat{A}_i \to 0$ → **that prompt contributes no
gradient.** The reward must *vary within the group* to produce a learning
signal. (This is also why the type/location sub-scores matter — they break ties
that a binary verdict alone would leave.)

---

## 3. The objective

Let the **token-level importance ratio** be

$$\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})}.$$

GRPO **maximizes**

$$
J(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\Bigg[\;
\underbrace{\min\!\Big(\rho_{i,t}\,\hat{A}_i,\; \operatorname{clip}(\rho_{i,t},\,1-\epsilon,\,1+\epsilon)\,\hat{A}_i\Big)}_{\text{PPO-clipped policy term}}
\;-\;
\beta\,\underbrace{\mathbb{KL}_{i,t}}_{\text{penalty}}
\;\Bigg]
$$

and the training loss is simply $L(\theta) = -\,J(\theta)$.

- The **policy term** is the standard PPO surrogate: it raises the log-prob of
  tokens with positive advantage and lowers it for negative advantage, with the
  clip $\epsilon$ preventing too-large per-step moves.
- The **$\beta\,\mathbb{KL}_{i,t}$ term** is the regularizer. It is **inside the
  loss**, not added to the reward.

---

## 4. The KL term — Schulman $k_3$ estimator, against the **frozen** reference

The penalty measures divergence of the current policy $\pi_\theta$ from a
**frozen reference** $\pi_{\text{ref}}$ — the SFT init (Arm-C `checkpoint-376`),
loaded as a separate model (this is the extra ~15 GB that only exists when
$\beta>0$). Per token, using the Schulman $k_3$ estimator:

$$\boxed{\;\mathbb{KL}_{i,t} = \frac{\pi_{\text{ref}}(o_{i,t})}{\pi_\theta(o_{i,t})} - \log\frac{\pi_{\text{ref}}(o_{i,t})}{\pi_\theta(o_{i,t})} - 1 \;\;\ge 0\;}$$

It is a low-variance, always-non-negative estimate of
$\mathrm{KL}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)$.

> **Two different "old" policies, don't confuse them:**
> the ratio $\rho$ is against $\pi_{\theta_{\text{old}}}$ (the policy that
> *generated this batch of rollouts*, for PPO importance sampling); the KL
> penalty is against $\pi_{\text{ref}}$ (the *frozen init*, fixed for the whole
> run). β leashes you to the init, not to last step.

---

## 5. Does β affect the weight update? **Yes — directly.**

This is the crux. Because the penalty is a term in the loss, it appears in the
gradient that the optimizer uses to update the weights:

$$
\nabla_\theta L \;=\; \underbrace{-\,\nabla_\theta(\text{policy term})}_{\text{push toward high-advantage tokens}}
\;+\; \beta\,\underbrace{\nabla_\theta\!\sum_{i,t}\mathbb{KL}_{i,t}}_{\text{pull }\pi_\theta\text{ back toward }\pi_{\text{ref}}}
$$

So **every gradient step is a tug-of-war**:

| force | direction |
|---|---|
| policy-gradient term | move weights to *increase reward* (raise prob of good tokens) |
| $\beta\,\nabla\mathbb{KL}$ | move weights to *stay near the frozen init* |

**β is literally the strength of the backward pull.** Bigger β → the
$\beta\nabla\mathbb{KL}$ component is larger → the optimizer is more reluctant to
move the weights away from the init → smaller effective policy change per step.
It backpropagates exactly like any other loss term.

### Why "in the loss" and not "as a reward bonus"
If instead you did $r_i \leftarrow r_i - \beta\,\text{KL}$ (reward bonus), the KL
would pass through the **group z-score normalization** of §2. A roughly
constant-per-group penalty largely **cancels** in $\hat{A}_i=(r_i-\text{mean})/\text{std}$,
so it would *not* reliably constrain the weights. Putting it in the loss makes it
a per-token gradient that acts **independently of the advantage**, every step.
(This is also why our inert `reasoning` reward — a near-constant ~0.5 — had no
effect on training: constants vanish under group normalization.)

---

## 6. What this means for our two runs (SFT-prompt, 1 epoch, from Arm-C 82.80/72.07)

Both runs kept KL **tiny (~0.02–0.03)** because the SFT init was already strong —
reward could be raised with only small weight moves, so $\pi_\theta$ never had to
wander far from $\pi_{\text{ref}}$.

| | β = 0.1 (strict) | β = 0.04 (loose) |
|---|---|---|
| $\beta\nabla\mathbb{KL}$ pull | stronger | weaker |
| sustained KL | ~0.020 | ~0.025 (drifts further) |
| final reward | 1.81 | **1.91** (moves more → earns more) |
| DS-MVTec eval | 81.72 (stable/rising) | 81.48 → **80.30** (drifts down) |

The looser β let the weights move more (higher KL, higher reward) — but the
extra reward was **over-fit to the (out-of-distribution Real-IAD) training
signal** and did not transfer to the MMAD benchmark, so DS eval drifted *down*.
That is the **reward-vs-eval decoupling / over-optimization**: more reward ≠
better model. Tightening β (0.1) resists weight movement → lower reward but
stable eval. Neither escaped the small-KL regime, consistent with neither
beating the SFT init.

Figure: `results/grpo_sftprompt_betacompare/reward_kl_curve.png` (reward + KL
vs step for both β).

---

## TL;DR
- Advantage = group z-score of reward; constant-within-group → no gradient.
- Loss = −(PPO-clipped policy term) **+** β·(per-token $k_3$ KL to the frozen init).
- **β is in the loss, so it is in the gradient, so it directly shapes the weight update** — it's a per-step pull back toward the init. Higher β = stronger leash = smaller moves. It does *not* touch the reward or the advantage.
