# RL Policy Optimization — Presentation Guide Part 1 (60 minutes)

## Scope of Part 1
This first session is intentionally detailed for a beginner audience.

It covers:
- RL foundations needed before policy optimization
- Why Monte Carlo appears in REINFORCE
- Policy gradient theorem intuition and derivation roadmap
- Value functions, advantage function, and their role in variance reduction
- Policy gradient variants (reward-to-go, baseline, vanilla PG)
- REINFORCE end-to-end (math + code mapping)
- Connection to RLHF and modern LLM finetuning (PPO)
- How to run and explain the repository code for REINFORCE

It does **not** cover A2C/A3C/PPO/TRPO in depth. That is in Part 2.

**Key reference**: [Cameron R. Wolfe — Policy Gradients: The Foundation of RLHF](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

---

## 1) Session objective
By the end of this Part 1 talk, the audience should be able to explain:
- What a policy is and why we optimize it directly
- What return means in episodic tasks
- Why Monte Carlo returns are unbiased but high variance
- How the log-derivative trick makes policy gradients computable
- How REINFORCE updates policy parameters
- What the advantage function is and why it matters for A2C/PPO
- How the four policy-gradient variants progressively reduce variance
- How theory maps to repository code
- Why policy gradients are the foundation of RLHF for LLMs

---

## 2) Suggested 60-minute agenda

- **0–8 min**: problem framing + what this app does
- **8–18 min**: RL fundamentals (MDP, trajectory, return)
- **18–28 min**: policy optimization objective + log-derivative trick
- **28–38 min**: policy gradient theorem + value/advantage functions
- **38–48 min**: detailed REINFORCE walkthrough + policy-gradient variants
- **48–55 min**: live run (single method + compare mode)
- **55–60 min**: Q&A + transition to Part 2

---

## 3) What this application is

This repository is a benchmarking app for policy-optimization methods on `CartPole-v1`.

### Core capabilities
- Standalone scripts per algorithm
- Unified orchestrator for algorithm comparison
- Standardized metrics and reproducible outputs
- Multi-run aggregation and report generation

### Entry points
- Unified orchestrator: [run_all_comparison.py](../run_all_comparison.py)
- Standalone REINFORCE runner: [policy_gradient_benchmark.py](../policy_gradient_benchmark.py)

### Core REINFORCE module for this session
- [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)

---

## 4) RL foundations (beginner-first)

### 4.1 MDP building blocks
An RL task is often modeled as a Markov Decision Process (MDP):
- **state** `sₜ` — what the agent observes
- **action** `aₜ` — what the agent does
- **reward** `rₜ` — immediate environment feedback
- **transition dynamics** `P(sₜ₊₁ | sₜ, aₜ)` — how the world changes
- **discount factor** `γ ∈ [0, 1]` — future-vs-immediate reward preference

### 4.2 Trajectory and return
One episode (trajectory) is:

```
τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...)
```

Discounted return from time `t`:

```
Gₜ = rₜ + γ · rₜ₊₁ + γ² · rₜ₊₂ + ...
   = Σₖ γᵏ · rₜ₊ₖ
```

Low-level intuition:
- Immediate rewards matter more when `γ < 1`.
- Return is the training signal indicating trajectory quality.

Two return forms (from Wolfe):
- **Finite-horizon undiscounted**: sum of rewards in fixed-length episodes
- **Infinite-horizon discounted**: discounted sum with `γ` for convergence

### 4.3 Why stochastic policies?
In policy optimization we model:

```
πθ(a | s)  =  P(action = a | state = s)
```

as a probability distribution over actions. This supports:
- exploration during training
- smooth gradients for optimization
- preference learning (not only hard action selection)

---

## 5) Value-Based vs Policy-Based

Two fundamentally different RL approaches:

| Aspect | Value-Based (DQN) | Policy-Based (This Series) |
|---------|-------------------|----------------------------|
| Strategy | Learn Q(s,a), choose argmax | Optimize π(a\|s) directly |
| Policy | Implicit (derived from Q) | Explicit (network outputs probabilities) |
| Actions | Primarily discrete | Discrete and continuous |
| Foundation | Bellman equation | Policy gradient theorem |
| Used in RLHF | No | Yes — PPO is policy optimization |

**Beginner analogy**:
- Value-based: score every menu dish, then pick top-scored
- Policy-based: learn your taste directly, without scoring all dishes

---

## 6) Policy optimization from first principles

### 6.1 Objective
We maximize expected trajectory return:

```
J(θ) = E_τ~πθ [ R(τ) ]
```

### 6.2 Gradient ascent
To maximize `J(θ)`, we use gradient ascent:

```
θ  ←  θ + α · ∇θ J(θ)
```

Each step:
1. Compute objective gradient wrt current parameters
2. Scale by learning rate `α`
3. Move parameters uphill (add, not subtract)

### 6.3 The hard part
Computing `∇J(θ)` is difficult because it includes `P(τ|θ)` (trajectory probability), which depends on unknown environment dynamics.

**Solution**: log-derivative trick.

---

## 7) The log-derivative trick (step by step)

### Step 1: The problem

```
∇J(θ) = ∇ Σ P(τ|θ) · R(τ)
```

But `P(τ|θ)` includes unknown environment dynamics.

### Step 2: Calculus identity

```
∇ log f(x)  =  ∇f(x) / f(x)
```

Rearranged:

```
∇f(x)  =  f(x) · ∇ log f(x)
```

### Step 3: Apply to ∇P(τ|θ)

```
∇P(τ|θ)  =  P(τ|θ) · ∇log P(τ|θ)
```

Now `P(τ|θ)` becomes a multiplier and the sum becomes an expectation we can sample.

### Step 4: Environment terms cancel
Trajectory probability factorization:

```
P(τ|θ) = P(s₀) · Π πθ(aₜ|sₜ) · P(sₜ₊₁|sₜ,aₜ)
```

After log:

```
log P(τ|θ) = log P(s₀) + Σ log πθ(aₜ|sₜ) + Σ log P(sₜ₊₁|sₜ,aₜ)
```

Taking `∇` wrt `θ`, only `πθ(a|s)` depends on `θ`; environment terms vanish.

### Step 5: Final form

```
∇J(θ) = E[ Σₜ ∇log πθ(aₜ|sₜ) · Gₜ ]
```

We only need:
1. Gradient of log-policy (from neural network/autograd)
2. Returns from sampled episodes

So this is model-free RL.

---

## 8) Policy gradient theorem — core equation

### Practical estimator

```
∇J(θ)  ≈  (1/N) Σᵢ [ Σₜ ∇log πθ(aₜ|sₜ) · Gₜ ]
```

### Term-by-term meaning

| Term | Meaning |
|------|---------|
| `∇log π(a\|s)` | direction to increase probability of chosen action in current state |
| `Gₜ` | quality of sampled outcome from that timestep |
| Product | good outcomes reinforce their actions, poor outcomes suppress them |
| `(1/N) Σ` | average over sampled trajectories |

### Intuition
- High-return actions become more probable
- Low-return actions become less probable
- Gradient points toward policies that make good trajectories more likely

That is why the algorithm is called **REINFORCE**.

### PyTorch form

```python
loss = -(log_probs * returns).sum()
loss.backward()       # autograd computes gradients
optimizer.step()      # minus sign turns descent into ascent objective
```

---

## 9) Value functions and advantage

### Four value functions (Wolfe)

| Function | Name | Meaning |
|----------|------|---------|
| `Vπ(s)` | State value | expected return from state `s` under policy `π` |
| `Qπ(s,a)` | Action value | expected return from taking `a` in `s`, then following `π` |
| `V*(s)` | Optimal state value | state value under best possible policy |
| `Q*(s,a)` | Optimal action value | action value under best policy; this is what DQN targets |

### Advantage function

```
A(s,a)  =  Q(s,a) - V(s)
```

“How much better is action `a` than the average action at state `s`?”

- `A > 0`: better-than-average action
- `A < 0`: worse-than-average action

### Why advantage matters
Using raw `G` is noisy because all actions share full-episode credit.
Using `A` reinforces only above-expected actions, reducing variance.

### Vanilla policy gradient (advantage form)

```
∇J(θ) = E[ ∇log π(a|s) · A(s,a) ]
```

This is what A2C/PPO optimize in practice.

### RLHF connection
PPO (used in ChatGPT RL phase) is advantage-based policy optimization with clipping and GAE.

**Evolution**: `REINFORCE → + baseline V(s) → advantage → A2C → PPO`

### Code mapping

```python
# benchmarks/a2c.py
advantage = returns - value_net(states)
policy_loss = -(log_probs * advantage)
```

---

## 10) Policy gradient variants — progressive variance reduction

### Variant 1: Basic REINFORCE

```
∇J = E[ Σₜ ∇log π(a|s) · R(τ) ]
```

Weight = full trajectory return. High variance.

### Variant 2: Reward-to-Go

```
∇J = E[ Σₜ ∇log π(aₜ|sₜ) · Gₜ ]
```

Weight = return from timestep `t` onward only. Lower variance.

> Implemented in `benchmarks/policy_gradient.py` via `_discounted_returns()`.

### Variant 3: Baseline subtraction

```
∇J = E[ Σₜ ∇log π · (Gₜ - b(s)) ]
```

Common baseline: `b(s)=V(s)`.

### Variant 4: Advantage form

```
∇J = E[ Σₜ ∇log π · Aπ(s,a) ]
```

Lowest variance among these variants.

> Implemented in `benchmarks/a2c.py` and `benchmarks/ppo.py`.

Key insight: same expected gradient target, different variance.

```
Lower variance → fewer episodes → faster convergence → more stable training
```

### GAE
GAE blends multi-step TD estimates with `λ` to trade off bias and variance. PPO uses GAE by default.

---

## 11) Monte Carlo in REINFORCE

### Meaning
REINFORCE waits until episode end, then computes full discounted returns from sampled rewards.
No bootstrap value target in base form.

### Why useful
- Unbiased return estimate for sampled policy
- Simple implementation

### Why hard
- High-variance updates
- Can be unstable/slower

### Numeric example
Rewards `[1,1,1]`, `γ=0.9`:

```
G₀ = 1 + 0.9×1 + 0.9²×1  = 2.71
G₁ = 1 + 0.9×1            = 1.90
G₂ = 1                    = 1.00
```

Earlier actions receive larger weight due to longer future influence.

### Return normalization

```python
G_norm = (G - mean(G)) / std(G)
```

Stabilizes gradient scale and acts as a simple baseline approximation.

### Reward-to-go trick
Use future-only rewards from each timestep (`G_t`), not episode-start total.
Reduces variance without bias.

---

## 12) REINFORCE algorithm (step-by-step)

```
1. Reset environment and sample one episode from current policy
2. Store log π(aₜ|sₜ) and reward rₜ at each step
3. At episode end compute discounted returns:
       Gₜ = rₜ + γ · Gₜ₊₁     (backward pass)
4. Optional return normalization:
       G = (G - mean) / std
5. Compute policy loss:
       L = -Σₜ log π(aₜ|sₜ) · Gₜ
6. Backprop and Adam update:
       loss.backward()
       optimizer.step()
7. Repeat for many episodes
```

Why minus sign? PyTorch optimizer does gradient descent by default; we need ascent on return.

---

## 13) Code mapping for Part 1

### REINFORCE implementation
- Config/dataclass: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Policy network (`PolicyNetwork`): 2-layer MLP (128 hidden units, ReLU), logits output
- Returns helper (`_discounted_returns`): backward `G = r + γ*G_next` reward-to-go
- Training loop (`run_policy_gradient`): trajectory sampling → returns → normalization → loss → backprop → optimizer step
- `PolicyGradientConfig`: `gamma=0.99, lr=1e-3, episodes=700, hidden_size=128, normalize_returns=True`

### Full script walkthrough (line-by-line)
- Detailed companion: [policy_gradient_line_by_line.md](policy_gradient_line_by_line.md)
- Source code: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)

### Theory-to-code correspondence

| Theory concept | Code location |
|----------------|---------------|
| `πθ(a\|s)` stochastic policy | `PolicyNetwork.forward()` → `Categorical(logits)` |
| `log π(a\|s)` | `dist.log_prob(action)` |
| `Gₜ` reward-to-go | `_discounted_returns()` |
| Baseline approximation | return normalization `(returns - mean) / std` |
| `L = -Σ log π · G` | `-(log_probs * returns).sum()` |
| `∇θ` autograd update | `loss.backward(); optimizer.step()` |

---

## 14) How the neural network learns policy in practice

Three phases repeated each episode:

### Phase 1 — Network outputs action probabilities

```python
logits = net(state_t)          # e.g., tensor([0.3, -0.8])
probs = torch.softmax(logits)  # e.g., tensor([0.75, 0.25])
```

At the beginning, random weights produce near-uniform probabilities.

### Phase 2 — Agent plays one full episode

```python
dist = Categorical(probs)
action = dist.sample()
log_probs.append(dist.log_prob(action))
```

Sampling (not hard argmax) provides intrinsic exploration.

### Phase 3 — Gradient updates weights

```python
returns = _discounted_returns(rewards, gamma)
returns_t = (returns_t - mean) / std

loss = -(log_probs * returns_t).sum()
loss.backward()
optimizer.step()
```

`log_prob × return` tells the network whether sampled actions should become more or less likely.

### Typical learning trajectory

```
Episode 1:    random weights → near-random actions     → reward ~20
Episode 50:   partially tuned → somewhat better         → reward ~80
Episode 300:  better calibrated → mostly correct actions → reward ~400
Episode 500:  converged → near-optimal policy           → reward 500
```

---

## 15) Can we do this without a neural network?

Yes. REINFORCE only requires:
1. A differentiable policy representation `π(a|s)`
2. Ability to compute `log π(a|s)` and gradients

### Option 1: Softmax table
For small discrete state spaces:

```python
theta = np.zeros((16, 4))

def policy(state):
    logits = theta[state]
    probs = np.exp(logits) / np.exp(logits).sum()
    return np.random.choice(4, p=probs)
```

### Option 2: Linear policy (no hidden layers)

```python
W = np.random.randn(n_actions, obs_dim) * 0.01
b = np.zeros(n_actions)

def policy(state):
    logits = W @ state + b
    probs = softmax(logits)
    return np.random.choice(n_actions, p=probs)
```

### Why neural network here?

| Representation | Advantage | Limitation |
|----------------|-----------|------------|
| Table | Exact and simple | Only small discrete state spaces |
| Linear | Fast and compact | Cannot capture nonlinear patterns |
| Neural network | Strong function approximation for continuous states | More tuning/compute |

CartPole states are continuous, so tabular representation is impractical. Neural networks generalize to unseen continuous states.

---

## 16) Exploration vs exploitation: Q-learning vs policy gradients

### Q-learning: explicit external exploration

```python
if random.random() < epsilon:
    action = env.action_space.sample()
else:
    action = np.argmax(Q[state])
```

Requires manual epsilon schedule.

### REINFORCE: intrinsic exploration

```python
probs = softmax(net(state))
action = Categorical(probs).sample()
```

Exploration is built into stochastic policy sampling.

### Typical evolution

```
Episode 1:    probs ≈ [0.52, 0.48]   → high exploration
Episode 100:  probs ≈ [0.70, 0.30]   → moderate exploration
Episode 300:  probs ≈ [0.85, 0.15]   → stronger exploitation
Episode 600:  probs ≈ [0.95, 0.05]   → near-greedy behavior
```

### Risk: premature deterministic collapse
If probabilities collapse too early (`[0.99, 0.01]`), exploration disappears.
A2C/A3C/PPO counter this with entropy bonus:

```python
entropy = dist.entropy().mean()
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

### Why entropy is needed in A2C/A3C (plain English)
- Entropy is a **“randomness bonus”** that keeps the policy from becoming overconfident too early.
- Without entropy, the agent may lock into one action pattern before it has explored enough.
- With entropy, the policy stays more exploratory early on, so it can discover better strategies.
- Practical tuning: higher entropy pressure is usually more useful early in training; later you can reduce `entropy_coef` so behavior becomes more decisive.

### Comparison summary

| Aspect | Q-Learning / DQN | REINFORCE | A2C / PPO |
|--------|-------------------|-----------|-----------|
| Mechanism | ε-greedy (external) | stochastic sampling (intrinsic) | sampling + entropy bonus |
| Controlled by | epsilon schedule | network weights | weights + `entropy_coef` |
| Schedule needed | Yes | No | No |

---

## 17) Demo commands (Part 1)

### Setup
```bash
uv sync
```

### REINFORCE only
```bash
uv run python policy_gradient_benchmark.py
```

### REINFORCE with custom hyperparameters
```bash
uv run python policy_gradient_benchmark.py --episodes 900 --gamma 0.99 --learning-rate 0.001
```

### REINFORCE with evaluation video
```bash
uv run python policy_gradient_benchmark.py --record-video --video-dir videos/policy_gradient --video-episodes 2
```

### REINFORCE through unified orchestrator
```bash
uv run python run_all_comparison.py --methods policy_gradient --policy-gradient-episodes 900
```

---

## 18) Teaching notes for low-level audience

Recommended board/slide sequence:
1. Define policy as action probabilities
2. Explain sampled trajectory as learning evidence
3. Show discounted return and early-action impact
4. Show policy gradient as “increase probability of actions with better outcomes”
5. Introduce `V(s)` and `Q(s,a)` as expectation tools
6. Introduce advantage `A = Q - V`
7. Walk variants: REINFORCE → reward-to-go → baseline → advantage
8. Clarify Monte Carlo trade-off: simple + unbiased, but noisy
9. Connect to RLHF: PPO = advantage-based variant + clipping
10. Explain 3-phase neural learning cycle
11. Clarify non-neural alternatives (table/linear) vs continuous-state need
12. Compare exploration approaches across methods

Common confusion to address:
- “Is this supervised learning?” → No labels; delayed reward signal
- “Why not always argmax?” → stochasticity is needed for exploration and unbiased PG sampling
- “Why normalize returns?” → stabilizes gradient magnitude
- “Difference between G and A?” → `G` raw return, `A` baseline-adjusted return
- “How does it explore without epsilon?” → sampling from policy distribution
- “Is a neural net always required?” → No; any differentiable policy representation works

---

## 19) Bridge to Part 2

Part 2 addresses the next question after REINFORCE:
- How to reduce variance and improve stability/performance further?

REINFORCE weaknesses → Part 2 methods:

| # | REINFORCE weakness | Part 2 method |
|---|--------------------|---------------|
| 1 | High-variance gradients | **A2C**: critic `V(s)` for advantage |
| 2 | Sample inefficiency | **A3C**: parallel workers and data decorrelation |
| 3 | Weak credit assignment | **PPO**: clipped surrogate + GAE |
| 4 | Hyperparameter sensitivity | **TRPO**: KL-constrained trust region |
| 5 | No update-size guardrail | **PPO**: clipped ratio prevents destructive updates |

### Quick preview: GRPO (for Part 2 discussion)

GRPO (Group Relative Policy Optimization) is a policy-optimization variant used often in modern LLM RL pipelines.

Plain-English intuition:
- Instead of relying mainly on a value critic, GRPO compares multiple sampled responses within the same prompt/context group.
- It learns from **relative quality** inside that group (better/worse than peers), then applies PPO-like clipped policy updates.

How it differs from PPO and TRPO:
- **GRPO vs PPO**: both use clipped updates, but PPO usually uses critic-based advantage (e.g., GAE), while GRPO emphasizes group-relative learning signals.
- **GRPO vs TRPO**: TRPO enforces an explicit KL trust-region constraint; GRPO keeps PPO-style clipping and group-relative signals, usually with simpler optimization.

All Part 2 methods are direct extensions of the policy-gradient foundations covered in this session.
