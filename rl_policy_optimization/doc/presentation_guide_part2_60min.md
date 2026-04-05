# RL Policy Optimization — Presentation Guide Part 2 (60 minutes)

## Scope of Part 2
This session covers the remaining policy optimization algorithms after REINFORCE:
- A2C
- A3C
- PPO
- TRPO

Focus:
- Why these methods improve stability/efficiency over vanilla REINFORCE
- How each method maps to repository code
- How to compare methods with reproducible benchmark workflow

---

## 1) Session objective
By the end of this Part 2 talk, the audience should understand:
- The role of critic baselines in reducing variance
- Why asynchronous rollout collection can help (A3C)
- Why clipped/trust-region updates stabilize policy learning (PPO/TRPO)
- How to run and compare all methods in this repository

---

## 2) Suggested 60-minute agenda

- **0–6 min**: quick recap from Part 1 (REINFORCE limitations)
- **6–20 min**: A2C (actor + critic)
- **20–30 min**: A3C (async workers)
- **30–43 min**: PPO (clipped surrogate + GAE)
- **43–50 min**: TRPO (trust-region concept + library integration)
- **50–57 min**: benchmark execution/comparison workflow
- **57–60 min**: recommendation summary + Q&A

---

## 3) A2C (Advantage Actor-Critic)

### Key intuition
REINFORCE uses full returns directly and can be noisy.
A2C introduces a value baseline `V(s)` and uses advantage:

```
Aₜ = Rₜ - V(sₜ)
```

This usually lowers gradient variance.

**Analogy (from HuggingFace Deep RL):** Imagine playing a video game. You're the **Actor** (you play and choose actions). Your friend sitting next to you is the **Critic** (they watch and tell you "that was a good move" or "that was terrible"). You don't know how to play at the beginning, so you try random actions. The Critic observes and provides feedback. Learning from that feedback, you update your strategy. Meanwhile, the Critic also gets better at judging over time.

### TD error as advantage estimator
In practice, computing the exact advantage `A = Q(s,a) - V(s)` requires two networks. A simpler approach uses the **TD error** as an approximation:

```
δₜ = rₜ + γ · V(sₜ₊₁) - V(sₜ)
```

This says: "was the actual reward + next state value better or worse than what I predicted for this state?" If δ > 0, the action was better than expected. This is a 1-step estimator of advantage — biased but much lower variance than full Monte Carlo returns.

### Code map
- Algorithm: [benchmarks/a2c.py](../benchmarks/a2c.py)
- Standalone runner: [a2c_benchmark.py](../a2c_benchmark.py)

### What to point out in code
- Shared backbone + policy/value heads
- Policy loss + value loss + entropy term
- Episode-wise returns and advantage computation

### Key concept → exact code anchors
- Model definition: `ActorCritic` in [../benchmarks/a2c.py](../benchmarks/a2c.py)
- Entrypoint: `run_a2c(...)` in [../benchmarks/a2c.py](../benchmarks/a2c.py)
- Advantage computation: `advantages_t = returns_t - values_t.detach()` in [../benchmarks/a2c.py](../benchmarks/a2c.py)
- Policy/value/entropy losses: `policy_loss`, `value_loss`, `entropy_t` in [../benchmarks/a2c.py](../benchmarks/a2c.py)

### How A2C works step by step

```
1. Collect one full episode with current policy
2. Compute Monte Carlo discounted returns Gₜ for each step
3. Get V(sₜ) from critic head for each state
4. Compute advantage: Aₜ = Gₜ - V(sₜ)
5. Compute combined loss:
     L = -Σ log π(a|s) · A        (policy — weighted by advantage)
       + value_coef · (V(s) - G)²  (critic — learn to predict returns)
       - entropy_coef · H(π)       (entropy — prevent collapse)
6. Backpropagate and update both heads jointly
7. Repeat for many episodes
```

### Theory-to-code correspondence

| Theory concept | Code in `benchmarks/a2c.py` |
|---------------|----------------------------|
| `π(a\|s)` — actor | `ActorCritic.policy_head` → `Categorical(logits)` |
| `V(s)` — critic | `ActorCritic.value_head` → scalar |
| `A = G - V(s)` — advantage | `advantage = returns - values.detach()` |
| Policy loss | `-(log_probs * advantage).sum()` |
| Value loss | `F.mse_loss(values, returns)` |
| Entropy bonus | `dist.entropy().mean()` |
| Combined loss | `policy_loss + 0.5*value_loss - 0.01*entropy` |

### What A2C fixes vs REINFORCE
- **Lower variance** — advantage baseline removes noise from raw returns
- **Better credit assignment** — only above-average actions get reinforced
- **Exploration protection** — entropy bonus prevents premature collapse
- **Still on-policy** — each trajectory used once, then discarded

---

## 4) A3C (Asynchronous Advantage Actor-Critic)

### Key intuition
A3C keeps actor-critic structure but collects experience using multiple workers in parallel.

**Why this matters (from Arthur Juliani / DeepMind 2016 paper):** In DQN, a single agent interacts with a single environment — experience is highly correlated (consecutive states are similar). DQN solves this with a replay buffer. A3C takes a completely different approach: instead of storing and replaying old experience, it **runs multiple agents in parallel**, each in their own environment copy. Since each worker explores from different states simultaneously, the batch of transitions collected is naturally decorrelated — **no replay buffer needed**.

**The landmark result:** In the original 2016 paper, A3C solved the same Atari games as DQN using just **16 CPU cores** instead of a powerful GPU — achieving better performance in **1 day** vs DQN's 8 days. The speedup is nearly linear: more workers → more diverse data → faster convergence.

Benefits:
- data decorrelation (replaces replay buffer)
- wall-clock speedup on CPU
- diverse exploration (each worker sees different states)

Trade-off:
- more systems complexity (processes, queues, synchronization)
- workers may have slightly stale parameters (policy lag)

### Code map
- Algorithm: [benchmarks/a3c.py](../benchmarks/a3c.py)
- Standalone runner: [a3c_benchmark.py](../a3c_benchmark.py)

### What to point out in code
- Worker loop collecting episodes and mini-rollouts
- Learner consuming batch queue
- Shared parameter refresh pattern

### Key concept → exact code anchors
- Worker rollout logic: `_worker_loop(...)` in [../benchmarks/a3c.py](../benchmarks/a3c.py)
- Shared parameters snapshot: `_snapshot_state_dict(...)` in [../benchmarks/a3c.py](../benchmarks/a3c.py)
- Learner loop and queue consumption: `run_a3c(...)` in [../benchmarks/a3c.py](../benchmarks/a3c.py)
- Async control signals: `data_queue`, `error_queue`, `stop_event` in [../benchmarks/a3c.py](../benchmarks/a3c.py)

### Architecture in detail

```
┌─────────────┐     ┌─────────────┐
│  Worker 1   │     │  Worker 2   │    ... N workers
│  (own env)  │     │  (own env)  │
│  collect    │     │  collect    │
│  rollouts   │     │  rollouts   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └──────┬────────────┘
              ▼
       ┌──────────────┐
       │ Shared Queue  │
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │   Learner     │
       │  apply grads  │
       │  update model │
       └──────────────┘
```

Each worker: runs its own env copy → collects `rollout_steps` transitions → computes advantages → pushes batch to queue.

Learner: dequeues batches → applies combined loss (same as A2C) → updates shared parameters.

Workers periodically refresh local weights from the shared model.

### Config defaults
`workers=4, rollout_steps=5, γ=0.99, lr=1e-3, value_coef=0.5, entropy_coef=0.01`

### A2C vs A3C
- **A2C** — synchronous updates, single process, easier to debug and reproduce
- **A3C** — async workers, multiprocessing, better wall-clock speed on multi-core CPUs
- **Trade-off**: multiprocessing adds engineering complexity (queues, sync, error handling)

---

## 5) PPO (Proximal Policy Optimization)

### Key intuition
PPO avoids overly large policy jumps with clipped objective:

```
L_clip(θ) = E[ min( rₜ(θ)·Aₜ , clip(rₜ(θ), 1-ε, 1+ε)·Aₜ ) ]

where rₜ(θ) = π_new(aₜ|sₜ) / π_old(aₜ|sₜ)
```

Often a practical default due to good stability/simplicity balance.

**The "cliff" analogy (from HuggingFace Deep RL):** Imagine standing on a mountain slope. Gradient ascent tells you to step right. A regular step is fine — you move toward the peak. But a slightly larger step sends you off a cliff into a completely different valley, and it takes forever to climb back. In supervised learning, other data points pull you back. In RL, **the data depends on your current policy** — if you take a bad step, your future data comes from a bad policy, creating a downward spiral. PPO prevents this by capping how big each step can be.

### The 6 clipping cases explained

The `min(unclipped, clipped)` formula creates 6 distinct behaviors depending on the ratio `r` and advantage `A`:

```
Case 1: r in [0.8, 1.2], A > 0  →  gradient pushes action UP (normal update)
Case 2: r in [0.8, 1.2], A < 0  →  gradient pushes action DOWN (normal update)
Case 3: r < 0.8,         A > 0  →  gradient pushes action UP (wants to recover)
Case 4: r < 0.8,         A < 0  →  gradient = 0 (already discouraged enough)
Case 5: r > 1.2,         A > 0  →  gradient = 0 (already encouraged enough)
Case 6: r > 1.2,         A < 0  →  gradient pushes action DOWN (wants to correct)
```

**Key insight:** In cases 4 and 5, the gradient is ZERO — the policy has already moved far enough in that direction, so the clip stops further movement. This is the mechanism that prevents catastrophic updates.

### Code map
- Algorithm: [benchmarks/ppo.py](../benchmarks/ppo.py)
- Standalone runner: [ppo_benchmark.py](../ppo_benchmark.py)

### What to point out in code
- Rollout buffer collection (1024 steps)
- GAE computation (`_compute_gae()`)
- Minibatch multi-epoch update with clipping

### Key concept → exact code anchors
- GAE function: `_compute_gae(...)` in [../benchmarks/ppo.py](../benchmarks/ppo.py)
- Core trainer: `run_ppo(...)` in [../benchmarks/ppo.py](../benchmarks/ppo.py)
- Clipped surrogate: `ratio`, `surr1`, `surr2`, `policy_loss` in [../benchmarks/ppo.py](../benchmarks/ppo.py)
- Multi-epoch minibatch update: inner `for start in range(...)` loop in [../benchmarks/ppo.py](../benchmarks/ppo.py)

### The clipping mechanism explained

The probability ratio measures how much the policy changed:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- `r = 1.0` → policy unchanged
- `r = 1.5` → action 50% more likely under new policy
- `r = 0.5` → action 50% less likely

Clipping to `[1-ε, 1+ε]` (default `[0.8, 1.2]`) caps how much any single update can push:

```python
# benchmarks/ppo.py — the core clipping
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantage
policy_loss = -torch.min(surr1, surr2).mean()
```

If advantage is positive (good action) and ratio > 1.2 → clip truncates the incentive to keep pushing.
If advantage is negative (bad action) and ratio < 0.8 → clip truncates the penalty.

### GAE (Generalized Advantage Estimation) — explained from scratch

GAE solves a fundamental problem: how do we estimate "was this action good?" There are two extreme answers, and GAE blends them.

**The problem: bias vs variance**

To compute advantage `A(s,a)`, we need to know how much return an action actually produced vs how much we expected. But we can estimate this at different horizons:

```
1-step TD:  δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)
            "Look 1 step ahead, then trust V for the rest"
            Low variance (one real reward) but high bias (V might be wrong)

Full MC:    Aₜ = Gₜ - V(sₜ)  =  (rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...) - V(sₜ)
            "Wait until episode ends, use all real rewards"
            No bias (all real rewards) but high variance (noisy trajectories)
```

**GAE: the best of both worlds**

GAE computes a weighted average of n-step advantages using parameter `λ`:

```
GAE(λ) = δₜ + (γλ)·δₜ₊₁ + (γλ)²·δₜ₊₂ + ...

where each δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)   (1-step TD error)
```

Think of it as: start with the 1-step estimate `δₜ`, then blend in corrections from future steps, but give exponentially decaying weight `(γλ)ᵏ` to each further step.

**What λ controls:**

```
λ = 0:     GAE = δₜ only          → pure 1-step TD (lowest variance, highest bias)
λ = 0.5:   GAE = blend of ~2-3 steps → moderate compromise
λ = 0.95:  GAE ≈ mostly Monte Carlo  → low bias, moderate variance (the sweet spot)
λ = 1:     GAE = full MC returns   → zero bias, highest variance
```

**Numerical example** — episode with 3 steps, `γ=0.99`, `λ=0.95`:

```
Rewards: [1, 1, 1],  V(s₀)=2.5, V(s₁)=1.8, V(s₂)=0.9, V(s₃)=0

δ₀ = 1 + 0.99·1.8 - 2.5 = 0.282     (step 0 was slightly above expected)
δ₁ = 1 + 0.99·0.9 - 1.8 = 0.091     (step 1 roughly as expected)
δ₂ = 1 + 0.99·0   - 0.9 = 0.100     (step 2 slightly above expected)

GAE₂ = δ₂                          = 0.100
GAE₁ = δ₁ + 0.99·0.95·GAE₂        = 0.091 + 0.094 = 0.185
GAE₀ = δ₀ + 0.99·0.95·GAE₁        = 0.282 + 0.174 = 0.456
```

Step 0 gets the highest advantage because it captures the accumulated above-average performance across all future steps.

**The code — walking backwards through the rollout:**

```python
# benchmarks/ppo.py AND benchmarks/trpo_native.py — _compute_gae()
def _compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):       # walk BACKWARDS
        not_done = 1.0 - float(dones[t])
        next_v = next_value if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + gamma * next_v * not_done - values[t]  # TD error
        gae = delta + gamma * gae_lambda * not_done * gae           # accumulate
        advantages[t] = gae
    returns = advantages + values                  # targets for value network
    return advantages, returns
```

The loop walks backwards because each `GAE[t]` depends on `GAE[t+1]` — exactly like computing discounted returns, but with the extra `λ` decay and TD errors instead of raw rewards.

**Where GAE appears in this repo:**
- PPO: `_compute_gae()` in `benchmarks/ppo.py` (λ=0.95)
- TRPO native: `_compute_gae()` in `benchmarks/trpo_native.py` (λ=0.97)
- A2C: does NOT use GAE — uses simpler `A = G - V(s)` with full MC returns

### Training flow
1. Collect rollout (1024 steps) with current policy, storing states, actions, log_probs, rewards, values
2. Compute advantages with GAE
3. Normalize advantages
4. Run 4 minibatch epochs (shuffle + split into batches of 64) on clipped loss
5. Discard rollout, repeat

### Config defaults
`rollout_steps=1024, update_epochs=4, minibatch_size=64, clip_eps=0.2, gae_lambda=0.95, lr=3e-4, value_coef=0.5, entropy_coef=0.01`

### Why PPO is the practical default
- **Stable** — clipping prevents catastrophic updates
- **Simple** — no constrained optimization (unlike TRPO)
- **Reuses data** — multiple epochs per rollout (more sample-efficient than A2C)
- **RLHF** — used to finetune ChatGPT, LLaMA, and other LLMs

---

## 6) TRPO (Trust Region Policy Optimization)

### Key intuition
TRPO constrains policy updates to remain in a trust region (typically KL-constrained).

**The mountain analogy (from Dilith Jayakody):** Imagine standing on a weird-shaped mountain. Gradient descent tells you to step right. A regular step moves you toward the valley (good). But a slightly larger step sends you into an entirely different pit — and recovering from it is very difficult. In supervised learning, other labeled data points pull you back. But in RL, **the data is non-stationary**: it depends on your current policy. If a bad update leads to bad actions, all future training data comes from those bad actions, creating a vicious cycle. TRPO prevents this by defining a "trust region" — the space around your current policy where you trust updates to be safe.

**Why RL is harder than supervised learning here:** In supervised learning, you always have the correct labels. Even if one gradient step is bad, the remaining labels correct it. In RL, if your policy takes a bad step, it generates bad trajectories, which produce bad gradients, which make the policy worse. The data distribution shifts with every update — this is the **non-stationarity** problem that makes large updates dangerous.

Pros:
- conservative, principled updates
- theoretical guarantee of monotonic improvement

Cons:
- heavier optimization machinery (conjugate gradient + backtracking line search)

### Code map
- Backend router (sb3 + native): [benchmarks/trpo.py](../benchmarks/trpo.py)
- Native TRPO trainer (discrete CartPole): [benchmarks/trpo_native.py](../benchmarks/trpo_native.py)
- Native TRPO math utilities (CG + line search): [benchmarks/trpo_core.py](../benchmarks/trpo_core.py)
- Standalone runner / backend flag: [trpo_benchmark.py](../trpo_benchmark.py)
- Unified orchestrator backend flag: [run_all_comparison.py](../run_all_comparison.py)

### What to point out in code
- Two implementation paths behind one interface: `backend="sb3"` vs `backend="native"`
- Native path implements trust-region mechanics explicitly (conjugate gradient + backtracking line search)
- Same benchmark output contract for both backends (`list[float]` rewards)
- Same CLI/orchestrator flow, only backend flag changes

### Key concept → exact code anchors
- Backend dispatch entrypoint: `run_trpo(...)` with `cfg.backend` in [../benchmarks/trpo.py](../benchmarks/trpo.py)
- Native trainer entrypoint: `run_trpo_native(...)` in [../benchmarks/trpo_native.py](../benchmarks/trpo_native.py)
- Native trust-region solve: `conjugate_gradients(...)` and `backtracking_line_search(...)` in [../benchmarks/trpo_core.py](../benchmarks/trpo_core.py)
- Native policy step internals: `fisher_vector_product`, `step_dir`, `full_step` in [../benchmarks/trpo_native.py](../benchmarks/trpo_native.py)
- CLI switch: `--backend sb3|native` in [../trpo_benchmark.py](../trpo_benchmark.py)
- Unified switch: `--trpo-backend sb3|native` in [../run_all_comparison.py](../run_all_comparison.py)

### The trust region constraint

```
maximize  E[ r(θ) · A ]
subject to  KL( π_old || π_new ) ≤ δ
```

KL divergence measures how different two probability distributions are. The constraint says: "improve the policy, but don't change it too much from the current one."

TRPO solves this exactly using conjugate gradient + line search. This is theoretically rigorous but computationally expensive per update.

### TRPO vs PPO
- **TRPO** — solves the constrained optimization exactly. Guaranteed monotonic improvement. Expensive.
- **PPO** — approximates the same idea with simple clipping. Much cheaper, nearly as stable, easier to implement.
- PPO was designed as a simpler alternative to TRPO. In practice, PPO is preferred for most tasks.

### Brief introduction to GRPO
GRPO (Group Relative Policy Optimization) is a newer policy-optimization variant, popular in RL workflows for LLMs.

Core idea in plain English:
- Instead of depending mostly on a value critic, GRPO compares responses inside a **group** of samples for the same prompt/context.
- Each response gets a **relative** learning signal (better/worse than peers), not only an absolute signal.
- It then applies PPO-style policy updates (ratio + clipping), but with this group-relative signal.

Intuition: it asks not only “was this action good?”, but “was it better than nearby alternatives in the same context?”

### GRPO vs PPO vs TRPO

| Aspect | GRPO | PPO | TRPO |
|---|---|---|---|
| Main training signal | Group-relative ranking/advantage | Advantage (usually critic + GAE) | Advantage with explicit KL constraint |
| Update-size control | PPO-style clipping | Clipping | KL trust-region constraint |
| Practical complexity | Medium (group sampling/ranking) | Low-medium | High |
| Typical use | Preference-comparison heavy RL (common in LLM RL) | General practical RL baseline | Cases prioritizing conservative/theoretical guarantees |

Quick summary:
- **TRPO**: most rigorous, most expensive.
- **PPO**: strongest practical default.
- **GRPO**: PPO-like stability with group-relative learning from comparisons.

### When to use TRPO
- Safety-critical systems or robotics where conservative updates are essential
- When you need a theoretical monotonic improvement guarantee
- When compute budget allows the extra cost per update

### Code: benchmarks/trpo.py
This repo now supports **parallel TRPO implementations** behind one benchmark interface:
- **SB3 backend** (`sb3-contrib`) for production-stable baseline behavior.
- **Native backend** (`benchmarks/trpo_native.py`) for educational/experimental comparison and full visibility into trust-region optimization internals.

For presentations, this is useful because you can show both:
- industrial wrapper integration (`sb3`), and
- algorithmic internals (`native`) using conjugate gradient + line search in project code.

### Demo commands for backend comparison (TRPO only)
```bash
uv run python trpo_benchmark.py --backend sb3 --timesteps 20000
uv run python trpo_benchmark.py --backend native --timesteps 20000
```

### Unified compare command (same pipeline, backend switch only)
```bash
uv run python run_all_comparison.py --methods trpo --trpo-backend sb3 --trpo-timesteps 20000
uv run python run_all_comparison.py --methods trpo --trpo-backend native --trpo-timesteps 20000
```

---

## 7) GRPO (Group Relative Policy Optimization)

### Key intuition
GRPO is the algorithm behind DeepSeek-R1 and the current frontier for RL-based LLM reasoning. It builds on PPO's clipped objective but **eliminates the critic network entirely**, replacing it with group-normalized rewards as the advantage estimate.

**Why this works for LLMs (from Cameron Wolfe):** PPO's critic exists to reduce variance in advantage estimation. But LLMs are extensively pre-trained models being finetuned — the high-variance problem is much less severe than in RL from scratch. Additionally, LLMs are mostly trained with **outcome rewards** (correct/incorrect), which makes per-token value estimation unnecessary. So the critic can be dropped entirely.

### How GRPO works step by step

```
1. For each prompt, generate G completions (a "group") from current policy
   E.g. G=8 different answers to one math problem

2. Score each completion with a reward rᵢ
   (e.g. 1 if correct, 0 if wrong — verifiable rewards, no reward model needed)

3. Compute group-relative advantage:
   Aᵢ = (rᵢ - mean(r)) / std(r)
   The group mean IS the baseline. No critic network V(s) needed.

4. Update policy with PPO-style clipping + KL divergence penalty against
   a reference policy to prevent drift:
   J_GRPO = E[ min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A) ] - β·KL(π_θ || π_ref)
```

### PPO vs GRPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Advantage | GAE over per-token V(s) | Normalized group reward |
| Critic | Required (large network) | None |
| Memory | Policy + critic + reward model (~3x) | Policy + reference only (~2x) |
| Reward source | Reward model (RLHF) | Verifiable rewards (RLVR) |
| Compute | Higher | ~50% less |
| Used in | ChatGPT, Claude (RLHF) | DeepSeek-R1, Qwen-3, OLMo-3 (reasoning) |

### RLHF vs RLVR

- **RLHF** (ChatGPT era): learned reward model from human preferences. Uses PPO. Risk: reward hacking.
- **RLVR** (DeepSeek-R1 era): verifiable rewards (correct/incorrect). No reward model needed. Uses GRPO. Harder to game.

### Where GRPO excels
- **Math reasoning** — DeepSeekMath: 46.8% → 51.7% on MATH benchmark
- **Code generation** — verifiable via test cases
- **Large reasoning models** — DeepSeek-R1, Qwen-3, OLMo-3
- **Accessibility** — HuggingFace TRL has built-in `GRPOTrainer`

### The evolution continues

```
REINFORCE → A2C → PPO → GRPO
(add critic)  (add clip)  (drop critic, group advantage)
```

### References
- Cameron R. Wolfe — GRPO: [cameronrwolfe.substack.com/p/grpo](https://cameronrwolfe.substack.com/p/grpo)
- DeepSeekMath paper: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1 paper: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

---

## 8) Side-by-side comparison narrative

| Method | Main idea | Strength | Common trade-off |
|---|---|---|---|
| A2C | Critic baseline + advantage | More stable than REINFORCE | Still on-policy and sample-hungry |
| A3C | Async multi-worker actor-critic | Better wall-clock behavior on CPU | Multiprocessing complexity |
| PPO | Clipped objective + repeated minibatch updates | Strong practical default | Hyperparameter-sensitive |
| TRPO | Trust-region constrained updates | Stable conservative improvement | Computationally heavier |

---

## 9) Demo commands (Part 2)

### Run each method
```bash
uv run python a2c_benchmark.py
uv run python a3c_benchmark.py
uv run python ppo_benchmark.py
uv run python trpo_benchmark.py
```

### Run all methods with one command
```bash
uv run python run_all_comparison.py --methods policy_gradient a2c a3c ppo trpo
```

### Run selected methods with videos
```bash
uv run python run_all_comparison.py --methods a2c ppo trpo --record-video --video-dir videos --video-episodes 2
```

### Example budget alignment override
```bash
uv run python run_all_comparison.py --methods a2c a3c ppo trpo --a2c-episodes 600 --a3c-episodes 600 --ppo-episodes 700 --trpo-timesteps 200000
```

---

## 10) Reporting and interpretation

Primary metrics produced by comparison runner:
- `max_avg_reward_100`
- `final_avg_reward_100`
- `elapsed_sec`
- `episodes`

Recommended interpretation:
- Compare peak capability (`max_avg_reward_100`)
- Compare end stability (`final_avg_reward_100`)
- Compare practical cost (`elapsed_sec`)
- Repeat across multiple seeds and aggregate

Aggregation pipeline:
- [../scripts/aggregate_results.py](../scripts/aggregate_results.py)
- [../scripts/generate_aggregate_report.py](../scripts/generate_aggregate_report.py)

---

## 11) Close and recommendations

Suggested practical baseline order:
1. Start with PPO.
2. Use A2C for simpler actor-critic baseline and faster iteration.
3. Use A3C when CPU parallel collection is desirable.
4. Use TRPO when conservative policy updates are required and extra compute is acceptable.

For foundational intuition, always anchor audience in Part 1 (policy optimization + Monte Carlo + REINFORCE) before deep-diving these variants.

---

## 12) Suggested sources used for this Part 2 narrative

- A2C: https://huggingface.co/blog/deep-rl-a2c
- A3C: https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
- PPO: https://huggingface.co/blog/deep-rl-ppo
- PPO (implementation patterns): https://docs.pytorch.org/rl/0.7/tutorials/multiagent_ppo.html
- TRPO: https://dilithjay.com/blog/trpo
- TRPO: https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9
- TRPO: https://towardsdatascience.com/trust-region-policy-optimization-trpo-explained-4b56bd206fc2/