# RL Hybrid Algorithms — Presentation Guide (60 minutes)

## Scope
This session covers three off-policy actor-critic algorithms for **continuous action spaces**:
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)

Focus:
- Why discrete-action methods (DQN) can't handle continuous control
- How actor-critic pairs replace argmax Q with learned policy
- How each algorithm fixes weaknesses of the previous one
- Theory-to-code mapping for every concept

---

## 1) Session objective
By the end of this talk, the audience should understand:
- Why continuous action spaces require a fundamentally different approach than DQN
- How DDPG extends Q-learning to continuous actions via a deterministic actor
- What three specific failure modes TD3 fixes in DDPG (and how)
- How SAC's entropy regularization provides built-in exploration and robustness
- How to map every concept to the repository code

---

## 2) Suggested 60-minute agenda

- **0–5 min**: quick recap — DQN and the discrete-action limitation
- **5–22 min**: DDPG (actor-critic for continuous, replay buffer, target networks)
- **22–37 min**: TD3 (three tricks: twin Q, delayed updates, target smoothing)
- **37–52 min**: SAC (entropy-regularized RL, stochastic policy, twin Q)
- **52–57 min**: comparison table + evolution line + practical recommendation
- **57–60 min**: Q&A

---

## 3) The Continuous Action Problem (Opening Hook)

### Why DQN can't do this

DQN works by computing Q(s,a) for every possible action, then taking the argmax. With discrete actions (left, right, jump), this is trivial — just compare 3 numbers.

But what if the action is a **continuous number**? For example:
- Robot joint torque: any value between -1.0 and +1.0
- Car steering angle: any value between -30° and +30°
- Chemical plant valve opening: any value between 0% and 100%

You can't enumerate infinite actions. Computing `argmax Q(s,a)` over a continuous range requires solving an optimization problem at every single step — too expensive.

### The DDPG solution

Instead of argmax, **learn a network** that directly outputs the best action:

```
DQN approach:   a* = argmax_a Q(s,a)     → impossible for continuous a
DDPG approach:  a* ≈ μ(s)               → actor network outputs action directly
```

The actor replaces argmax. The critic Q(s,a) evaluates the actor's choices. This is the core idea behind all three algorithms in this session.

---

## 4) DDPG (Deep Deterministic Policy Gradient)

### Key intuition
DDPG is "deep Q-learning for continuous action spaces." It learns two networks simultaneously:

```
Actor μθ(s):    state → continuous action (deterministic)
Critic Qϕ(s,a): state + action → Q-value
```

The critic learns via Bellman MSE (like DQN). The actor learns by gradient ascent on Q — adjusting its output to maximize the critic's evaluation.

### Three key mechanisms

**1. Replay Buffer** — stores (s,a,r,s',done) transitions. Training samples random minibatches. Off-policy: reuses old data, unlike on-policy methods (A2C, PPO).

**2. Target Networks** — slow-moving copies of actor and critic, updated via Polyak averaging:

```
θ' ← τ·θ + (1-τ)·θ'     (τ = 0.001 in this repo)
```

Why needed: if the critic target depends on the same network being trained, learning is unstable (the target moves with every update). Polyak averaging provides a slowly-moving target.

**3. Exploration Noise** — the actor outputs a deterministic action. Without noise, the agent would always take the same action in the same state and never explore. Adding Gaussian or OU noise to actions during training provides exploration:

```
a = μ(s) + noise    (training)
a = μ(s)             (evaluation — no noise)
```

### Training loop step by step

```
1. Act:     a = μ(s) + noise, execute in environment
2. Store:   (s, a, r, s', done) in replay buffer
3. Sample:  random minibatch from buffer
4. Critic:  target = r + γ · Q'(s', μ'(s'))
            loss_critic = MSE(Q(s,a), target)
5. Actor:   loss_actor = -Q(s, μ(s)).mean()
            "find the action that makes the critic happiest"
6. Targets: Polyak update both target networks
```

### The actor update explained

This is the most important line to understand:

```python
# rl_continuous_optimization/benchmarks/ddpg.py — actor update
actor_loss = -critic(s, actor(s)).mean()
```

The gradient flows: actor → actions → critic → Q-value. The minus sign means "maximize Q." The actor adjusts its output so the critic gives higher Q-values. This replaces the impossible `argmax_a Q(s,a)` with a differentiable approximation.

### Known weaknesses (motivation for TD3 and SAC)

1. **Q-value overestimation** — the single critic develops false peaks in Q-landscape. The actor exploits these errors, leading to brittle behavior that can collapse suddenly.
2. **Hyperparameter sensitive** — noise scale, learning rates, buffer size, and Polyak rate all matter a lot. Small changes can break training.
3. **Deterministic policy** — no built-in exploration mechanism. Relies entirely on added noise, which must be manually tuned.

### Code map
- Benchmark runner: [rl_continuous_optimization/ddpg_benchmark.py](../rl_continuous_optimization/ddpg_benchmark.py)
- Shared implementation: [rl_continuous_optimization/benchmarks/ddpg.py](../rl_continuous_optimization/benchmarks/ddpg.py)
- Replay buffer: [rl_continuous_optimization/benchmarks/replay_buffer.py](../rl_continuous_optimization/benchmarks/replay_buffer.py)
- Unified comparison runner: [rl_continuous_optimization/run_all_comparison.py](../rl_continuous_optimization/run_all_comparison.py)

### Theory-to-code correspondence

| Theory concept | Code location |
|---------------|--------------|
| Actor μθ(s) | `Actor` in `benchmarks/ddpg.py` |
| Critic Qϕ(s,a) | `Critic` in `benchmarks/ddpg.py` |
| Replay buffer | `ReplayBuffer` in `benchmarks/replay_buffer.py` |
| Target networks | `actor_target`, `critic_target` in `benchmarks/ddpg.py` |
| Polyak update | in-place soft update loop with `tau` |
| Exploration noise | Gaussian noise added to actor action |
| Critic loss (Bellman MSE) | `F.mse_loss(current_q, target_q)` |
| Actor loss (maximize Q) | `-critic(s, actor(s)).mean()` |
| Benchmark orchestration | `run_timed` and `run_all_comparison.py` |

### Config defaults

```
env=Pendulum-v1, episodes=160, max_steps=200
γ=0.99, actor_lr=1e-3, critic_lr=1e-3
batch_size=256, replay=200K, warmup_steps=5000
polyak τ=0.005
```

---

## 5) TD3 (Twin Delayed Deep Deterministic)

### Key intuition
TD3 identifies three specific failure modes of DDPG and adds one targeted fix for each. Nothing else changes — it's still a deterministic actor-critic with replay buffer and target networks.

### The problem: Q-value overestimation

DDPG's single critic tends to overestimate Q-values. Here's why this is catastrophic:

```
Critic develops false peak → actor exploits it →
actor outputs bad actions → environment gives bad rewards →
but actor keeps exploiting the false peak → training collapses
```

### Trick 1: Clipped Double-Q Learning

Train **two** independent Q-networks. Use the **minimum** for computing targets:

```
y = r + γ · min(Q₁_targ, Q₂_targ)(s', a')
```

Why it works: if one critic overestimates, the other likely doesn't. Taking the min suppresses overestimation bias.

### Trick 2: Delayed Policy Updates

Update the actor **less frequently** — once every 2 Q-updates:

```
if step % policy_delay == 0:
    update actor
    update target networks
```

Why it works: let Q-functions stabilize before the actor adjusts.

### Trick 3: Target Policy Smoothing

Add **clipped noise** to the target action:

```
ε ~ N(0, σ), clipped to [-c, c]
a' = clip(μ_targ(s') + ε)
```

Why it works: smooths Q-landscape around target actions, preventing the policy from exploiting narrow Q-function spikes.

### Combined target equation

```
ε ~ N(0, 0.2), clipped to [-0.5, 0.5]
a' = clip(μ_targ(s') + ε, a_low, a_high)
y  = r + γ(1-d) · min(Q₁_targ, Q₂_targ)(s', a')
```

Policy maximizes Q₁ only.

### What TD3 changes vs DDPG

| Aspect | DDPG | TD3 |
|--------|------|-----|
| Critics | 1 | 2 (twin) |
| Target Q | Q'(s',μ'(s')) | min(Q₁',Q₂')(s', μ'(s')+noise) |
| Policy update | every step | every 2nd step |
| Target action | clean | smoothed with clipped noise |

### Code implementation notes

The final implementation includes a dedicated TD3 module and benchmark runner:
- [rl_continuous_optimization/benchmarks/td3.py](../rl_continuous_optimization/benchmarks/td3.py)
- [rl_continuous_optimization/td3_benchmark.py](../rl_continuous_optimization/td3_benchmark.py)

Key training differences in code:

```python
# Change 1: twin critics
current_q1, current_q2 = critic(s, a)

# Change 2: min target + smoothing
noise = (torch.randn_like(a) * policy_noise * max_action).clamp(-noise_clip * max_action, noise_clip * max_action)
next_action = (actor_target(ns) + noise).clamp(-max_action, max_action)
target_q = r + nd * gamma * torch.min(target_q1, target_q2)

# Change 3: delayed update
if update_step % policy_delay == 0:
    actor_loss = -critic.q1_only(s, actor(s)).mean()
```

---

## 6) SAC (Soft Actor-Critic)

### Key intuition
SAC maximizes both return **and** entropy (randomness):

```
J = E[Σ rₜ + α · H(π(·|sₜ))]
```

The agent is rewarded for exploring widely while still getting high return.

### Why entropy matters

Without entropy (DDPG/TD3): the policy converges quickly to a single action per state. If it converges to the wrong action, it's stuck.

With entropy (SAC): the policy maintains a distribution. Even after training, it samples from a range, providing built-in exploration, robustness to local optima, and multi-modal behavior.

### Stochastic policy (the key difference)

```
DDPG/TD3: a = μ(s)                    (deterministic)
SAC:      mean, log_std = actor(s)     (stochastic)
          a ~ Normal(mean, exp(log_std))
```

### Three differences from TD3

1. **Stochastic policy** — samples from learned distribution, no external noise needed
2. **Entropy in Q-target** — `-α log π(a'|s')` added to target
3. **No explicit target smoothing** — stochasticity provides it naturally

### Q-target with entropy

```
ã' ~ π(·|s')     ← sample from CURRENT policy (not target!)
y = r + γ(1-d) · [min(Q₁,Q₂)(s',ã') - α log π(ã'|s')]
```

### Temperature α

```
High α: more exploration | Low α: more exploitation | α→0: deterministic
```

Fixed (this repo: α=0.2) vs auto-tuned (preferred in practice).

### Code walkthrough

```python
# rl_continuous_optimization/benchmarks/sac.py

# TwinQ update
current_q1, current_q2 = critic(s, a)
critic_loss = MSE(current_q1, target) + MSE(current_q2, target)

# Target with entropy term
next_a, next_logp = actor.sample(ns, deterministic=False)
target_q = min(target_q1, target_q2) - alpha * next_logp
target = r + nd * gamma * target_q

# Actor update
pi_action, logp_pi = actor.sample(s, deterministic=False)
actor_loss = (alpha * logp_pi - min_q_pi).mean()

# Target update (twin critic target)
for p, tp in zip(critic.parameters(), critic_target.parameters()):
    tp.data.mul_(1.0 - tau).add_(tau * p.data)
```

### Theory-to-code correspondence

| Theory concept | Code location |
|---------------|--------------|
| Actor π(a\|s) | `GaussianActor` in `benchmarks/sac.py` |
| Twin Q₁,Q₂ | `TwinCritic` in `benchmarks/sac.py` |
| Entropy α | `SACConfig.alpha` (default 0.2) |
| Action squashing | tanh + log-prob correction in `actor.sample()` |
| Replay buffer | `ReplayBuffer` in `benchmarks/replay_buffer.py` |
| Train script | `rl_continuous_optimization/sac_benchmark.py` |
| Unified comparison | `rl_continuous_optimization/run_all_comparison.py` |

### Architecture note

This implementation uses SAC v2 style (TwinQ + stochastic actor, no separate V network) with fixed α.

---

## 7) Comparison and Evolution

| Feature | DDPG | TD3 | SAC |
|---------|------|-----|-----|
| Policy type | deterministic | deterministic | stochastic |
| Q-networks | 1 | 2 (twin) | 2 (twin) |
| Exploration | added noise | added noise | built-in (entropy) |
| Overestimation fix | none | clipped double-Q | clipped double-Q |
| Policy update freq | every step | delayed (1:2) | every step |
| Target smoothing | none | explicit (noise) | implicit (stochastic) |
| Hyperparam sensitivity | high | moderate | low |

### Evolution

```
DQN (discrete, 2015) → DDPG (continuous, 2016) → TD3 (fix overest., 2018) → SAC (entropy, 2018)
```

### Recommendation

1. Start with SAC — most robust default
2. Try TD3 when SAC's entropy hinders
3. Use DDPG only for learning or baseline

---

## 8) Exploration comparison: Policy Optimization vs Hybrid

| Aspect | A2C/PPO (on-policy) | DDPG/TD3 (off-policy) | SAC (off-policy) |
|--------|--------------------|-----------------------|------------------|
| Policy | stochastic | deterministic | stochastic |
| Exploration | sampling + entropy | external noise | sampling + entropy |
| Data reuse | once (discard) | replay buffer | replay buffer |
| Action space | discrete or cont. | continuous only | continuous only |

SAC bridges both paradigms: stochastic exploration (like PPO) + off-policy replay (like DDPG).

---

## 9) Demo commands

```bash
cd rl_continuous_optimization

# Run individual algorithms
uv run python ddpg_benchmark.py --episodes 160
uv run python td3_benchmark.py --episodes 160
uv run python sac_benchmark.py --episodes 160

# Run all methods together
uv run python run_all_comparison.py --methods ddpg td3 sac --output-dir outputs

# Aggregate multi-seed outputs and create report
uv run python scripts/aggregate_results.py --inputs outputs/*.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Continuous Control Aggregate Report"
```

---

## 10) Teaching notes for low-level audience

Recommended sequence:
1. Start with the continuous action problem: "you can't argmax over infinity"
2. Introduce actor as the "argmax replacement"
3. Draw actor-critic loop: state → actor → action → env → reward → critic
4. Explain replay buffer: "memory of past experiences, randomly sampled"
5. Explain target networks: "slow copy to stabilize training"
6. DDPG weaknesses: Q overestimation → policy exploits errors
7. TD3 tricks: twin Q (min), delayed updates, target smoothing
8. SAC shift: "what if the agent is rewarded for being uncertain?"
9. Entropy = randomness bonus. More entropy = more exploration
10. Compare all three side by side

Common confusion:
- "Why not discretize?" → loses precision, exponential bins for multi-dim
- "Why the minus sign in actor loss?" → PyTorch minimizes; minus = maximize
- "A2C critic vs DDPG critic?" → A2C: V(s) baseline; DDPG: Q(s,a) for actor gradient
- "Why twin Q if SAC is stochastic?" → stochasticity smooths but doesn't fix overestimation

---

## 11) Suggested sources

- DDPG: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-4643c1f71b2e/
- DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
- DDPG: https://medium.com/@amaresh.dm/how-ddpg-deep-deterministic-policy-gradient-algorithms-works-in-reinforcement-learning-117e6a932e68
- DDPG: https://intellabs.github.io/coach/components/agents/policy_optimization/ddpg.html
- DDPG: https://www.nature.com/articles/s41598-025-99213-3
- TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
- TD3: https://medium.com/@heyamit10/twin-delayed-deep-deterministic-policy-gradient-td3-fc8e9950f029
- TD3: https://discovery.ucl.ac.uk/id/eprint/10210972/1/A%20TD3-Based%20Reinforcement%20Learning%20Algorithm%20with%20Curriculum%20Learning%20for%20Adaptive%20Yaw%20Control%20in%20All-Wheel-Drive%20Electr.pdf
- SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
- SAC: https://towardsdatascience.com/navigating-soft-actor-critic-reinforcement-learning-8e1a7406ce48/