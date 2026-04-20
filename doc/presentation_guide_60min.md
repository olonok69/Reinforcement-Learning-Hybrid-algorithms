# RL Continuous Control — Presenter Guide (60 minutes)

> **Session 6 of the RL course.** Previous sessions covered MDPs, Q-learning, DQN, policy gradient,
> and A2C/PPO. The audience is comfortable with value functions, neural networks, and the basic
> policy-gradient theorem. Build on that — no need to re-explain Bellman equations from scratch.

---

## Title note

The slide deck is titled **"Hybrid Algorithms, the best from both worlds: DDPG, SAC & TD3"**.

The subtitle *"best from both worlds"* is conceptually correct: all three algorithms are
**actor-critic** methods that marry a **policy-gradient actor** (world 1) with a
**value-based critic** (world 2). That framing is worth making explicit on the opening slide.

However, the word *"Hybrid"* as a primary label is non-standard. In the broader RL literature
"hybrid" usually implies combining model-free RL with model-based planning, or RL with
supervised learning. A more precise and searchable title would be:

> **RL Continuous Control: Off-Policy Actor-Critic Methods — DDPG, TD3 & SAC**
> *Combining the best of policy-gradient and value-based learning*

**Recommendation:** keep the "best from both worlds" tagline as a subtitle, but lead with
"Continuous Control" or "Off-Policy Actor-Critic" so the audience immediately maps it to the
correct literature cluster. If the broader course uses "Hybrid Algorithms" as a recurring
theme for actor-critic, that is a valid in-course convention — just state it explicitly in the
opening minute.

---

## Scope

Three off-policy actor-critic algorithms for **continuous action spaces**:

| Algorithm | Key idea | Introduced |
|-----------|----------|-----------|
| DDPG | Deterministic actor + single critic + target nets | Lillicrap et al. 2015 |
| TD3  | Twin critics + delayed actor + target smoothing | Fujimoto et al. 2018 |
| SAC  | Stochastic actor + entropy regularisation + twin critics | Haarnoja et al. 2018 |

Repository layout used during this session:

```
rl_continuous_optimization/
├── benchmarks/          ← all algorithm implementations + replay buffer
│   ├── ddpg.py          ← DDPG: Actor, Critic, run_ddpg()
│   ├── td3.py           ← TD3:  Actor, TwinCritic, run_td3()
│   ├── sac.py           ← SAC:  GaussianActor, TwinCritic, run_sac()
│   ├── replay_buffer.py ← shared off-policy experience store
│   └── common.py        ← BenchmarkResult, run_timed(), metrics helpers
├── ddpg_benchmark.py    ← standalone CLI for DDPG
├── td3_benchmark.py     ← standalone CLI for TD3
├── sac_benchmark.py     ← standalone CLI for SAC
├── run_all_comparison.py← unified multi-method runner
└── scripts/
    ├── aggregate_results.py      ← compute mean/std across seeds
    └── generate_aggregate_report.py ← HTML/text report generation
```

Concept docs: [`01_ddpg.md`](01_ddpg.md) · [`02_td3.md`](02_td3.md) · [`03_sac.md`](03_sac.md)  
Code guide: [`../benchmarks/README.md`](../benchmarks/README.md)

---

## Learning objectives

By the end of the session the audience should be able to:

1. Explain why the DQN-style `argmax_a Q(s,a)` does not scale to continuous action spaces.
2. Trace the full DDPG loop: actor produces action → critic evaluates → gradient flows back to actor.
3. Name TD3's three fixes and explain *why* each one helps stability.
4. Explain what the entropy term in SAC does and how `alpha` controls the exploration-exploitation tradeoff.
5. Read the benchmark code and identify the exact lines that implement each key idea.
6. Run a reproducible comparison of the three algorithms and interpret the output metrics.

---

## Detailed 60-minute flow

---

### 0–5 min — The continuous action problem and the DQN wall

**Slide focus:** motivate why we need a new family of algorithms.

**Talking points:**

- DQN selects actions via `argmax_a Q(s,a)`. That `argmax` is computed by enumeration or a forward
  pass over a discrete output head. With a robot arm that has 7 joints each moving in `[-1, 1]`
  the search space is uncountably infinite — enumeration is impossible.
- Discretising a continuous action space loses precision and explodes combinatorially:
  10 joints × 100 bins each = 10^20 combinations.
- The solution introduced in DDPG: **learn the argmax directly** as a separate parametric function
  (the actor `mu(s; theta)`). This is the core actor-critic idea applied to continuous control.
- Connect to previous sessions: "We used policy gradient in PPO to learn a *stochastic* policy.
  DDPG learns a *deterministic* policy and evaluates it with a Q-function critic."

**Anticipated question:** *"Why not just use PPO with a Gaussian policy head on continuous actions?"*  
Answer: You can (and it works). DDPG/TD3/SAC are **off-policy** — they reuse past experience via a
replay buffer, which makes them sample-efficient for slow simulators or real-robot settings. PPO
is on-policy and discards old data.

---

### 5–20 min — DDPG: the actor-critic blueprint for continuous control

**Slide focus:** the four components of DDPG and how they interact.

#### 5–9 min — Architecture overview

**Talking points:**

- Four networks, always in two pairs:
  - **Online actor** `mu(s; theta)` + **target actor** `mu(s; theta')`  
  - **Online critic** `Q(s,a; phi)` + **target critic** `Q(s,a; phi')`
- Target networks exist solely to stabilise the Bellman target — they evolve slowly via soft
  (Polyak) update. Without them the network is chasing a moving target with its own gradients
  and training diverges.
- **Replay buffer** decouples data collection from learning: any past transition `(s,a,r,s')` can
  be sampled, breaking the temporal correlation that breaks SGD assumptions.

**Code pointer — network definitions:**  
[`benchmarks/ddpg.py` lines 35–63](../benchmarks/ddpg.py)

```python
# Actor: state -> tanh(net(state)) * max_action
# tanh squashes output to (-1,1); max_action rescales to env bounds
class Actor(nn.Module):
    def forward(self, state):
        return torch.tanh(self.net(state)) * self.max_action

# Critic: concatenate (state, action) -> scalar Q-value
class Critic(nn.Module):
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))
```

Point out: the critic takes `(s, a)` as a joint input. This is only possible because at training
time we query it with the actor's chosen action — not over all possible actions.

**Code pointer — network instantiation and target copy:**  
[`benchmarks/ddpg.py` lines 77–83](../benchmarks/ddpg.py)

```python
actor_target.load_state_dict(actor.state_dict())   # start identical
critic_target.load_state_dict(critic.state_dict())  # start identical
```

#### 9–14 min — Warmup and exploration

**Talking points:**

- Off-policy algorithms need a pre-filled buffer before the first gradient step. The first
  `warmup_steps=5000` environment steps use *random actions* to seed diverse experience.
- During training, exploration is additive Gaussian noise on top of the deterministic action:
  `a = clip(mu(s) + N(0, sigma), low, high)`  
  DDPG has no intrinsic exploration mechanism — the designer chooses `sigma`.
- This is a known weakness: if the noise is poorly tuned, the agent either over-explores or
  gets stuck in a local optimum.

**Code pointer — warmup and noise:**  
[`benchmarks/ddpg.py` lines 101–108](../benchmarks/ddpg.py)

```python
if global_step < cfg.warmup_steps:
    action = env.action_space.sample()       # pure random
else:
    action = actor(state_t).cpu().numpy()    # deterministic actor
    noise = np.random.normal(0.0, cfg.exploration_noise * max_action, ...)
    action = np.clip(action + noise, ...)    # clamp to env bounds
```

Config reference: [`DDPGConfig`](../benchmarks/ddpg.py) — `warmup_steps=5000`,
`exploration_noise=0.1`.

#### 14–20 min — The update equations

**Talking points:**

Step through the update in the order it executes in the code:

1. **Sample a mini-batch** `(s, a, r, s', done)` from the replay buffer.
2. **Compute Bellman target** (with target networks, no gradient):  
   `y = r + γ·(1-done)·Q_target(s', μ_target(s'))`
3. **Update critic** — minimise MSE between online Q-estimate and target:  
   `L_critic = MSE( Q(s,a), y )`
4. **Update actor** — maximise Q via gradient ascent, implemented as minimising the negative:  
   `L_actor = -(1/N)·Σ Q(s, μ(s))`
5. **Soft update** both target networks (Polyak averaging, `tau=0.005`):  
   `θ' ← (1-τ)·θ' + τ·θ`

**Code pointer — full update block:**  
[`benchmarks/ddpg.py` lines 118–144](../benchmarks/ddpg.py)

```python
# 2-3: critic update
next_action = actor_target(ns)
target_q = critic_target(ns, next_action)
target_q = r + nd * cfg.gamma * target_q      # nd = (1 - done)
critic_loss = F.mse_loss(critic(s, a), target_q)
critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

# 4: actor update — gradient flows through critic into actor
actor_loss = -critic(s, actor(s)).mean()
actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

# 5: Polyak soft update of both target pairs
for p, tp in zip(critic.parameters(), critic_target.parameters()):
    tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)
```

**Key insight to highlight:** the actor is updated by *differentiating through* the critic.
The gradient chain is: `actor_loss → Q(s,a) → actor(s) → actor.parameters()`. This is the
deterministic policy gradient theorem (Silver et al. 2014).

**Concept doc:** [`01_ddpg.md`](01_ddpg.md)  
**Known risks:** single critic → Q overestimation accumulates → actor chases inflated values →
instability. This motivates TD3.

---

### 20–35 min — TD3: three surgical fixes to DDPG

**Slide focus:** show DDPG code, then diff against TD3 code to make the improvements concrete.

**Talking points (intro):**  
Fujimoto et al. 2018 diagnosed three failure modes in DDPG and added one fix for each. The
changes are small in code but large in impact.

**Concept doc:** [`02_td3.md`](02_td3.md)

#### Fix 1 — Twin critics (lines 47–73 of td3.py)

**Talking points:**

- Single-critic DDPG overestimates Q because the actor always pushes toward the maximum of Q.
  With noise and approximation errors this bias compounds.
- TD3 maintains **two independent critics** `Q1` and `Q2`. The Bellman target uses their
  **minimum**, not either value alone:
  `y = r + γ·(1-done)·min( Q1_target(s',a'), Q2_target(s',a') )`
- Min-of-two is a pessimistic estimator, but pessimism is safer than optimism when the actor
  is continually exploiting the critic's errors.

**Code pointer — TwinCritic and clipped double-Q target:**  
[`benchmarks/td3.py` lines 47–73](../benchmarks/td3.py) and
[lines 119–121](../benchmarks/td3.py)

```python
class TwinCritic(nn.Module):
    # q1 and q2 are independent MLP heads, same architecture as DDPG Critic
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
    def q1_only(self, state, action):     # used for actor update
        return self.q1(torch.cat([state, action], dim=1))

# target: min of both critics
target_q1, target_q2 = critic_target(ns, next_action)
target_q = r + nd * cfg.gamma * torch.min(target_q1, target_q2)  # clipped!
```

#### Fix 2 — Delayed actor updates (`policy_delay`)

**Talking points:**

- In DDPG the actor and critic update every step. But if the critic is inaccurate (early
  training) the actor gradient is noisy and can push the policy in a wrong direction.
- TD3 updates the actor only every `policy_delay=2` critic updates, giving the critic time to
  converge first.
- Also note: when the actor does update, the target networks update at the same time.
  This keeps the full system in sync.

**Code pointer — delayed actor gate:**  
[`benchmarks/td3.py` lines 122–133](../benchmarks/td3.py)

```python
update_step += 1

# critics update every step
critic_loss = ...
critic_opt.step()

# actor (and targets) update only every policy_delay steps
if update_step % cfg.policy_delay == 0:
    actor_loss = -critic.q1_only(s, actor(s)).mean()
    actor_opt.step()
    # soft update targets here (inside the if)
```

#### Fix 3 — Target policy smoothing

**Talking points:**

- Without smoothing, the target actor always produces the same deterministic action for a given
  `s'`. If the critic has a sharp peak at that action, the target Q value is artificially high.
- TD3 adds **clipped Gaussian noise** to the target action, effectively averaging the target Q
  over a small neighbourhood of actions — regularising the Q surface:
  ```
  a' = clip( μ_target(s') + ε,  -a_max, a_max )
  ε  = clip( N(0, σ),  -c, c )
  ```

**Config values:** `policy_noise=0.2`, `noise_clip=0.5`.

**Code pointer — target smoothing noise:**  
[`benchmarks/td3.py` lines 113–121](../benchmarks/td3.py)

```python
noise = (torch.randn_like(a) * cfg.policy_noise * max_action).clamp(
    -cfg.noise_clip * max_action,
    +cfg.noise_clip * max_action,
)
next_action = (actor_target(ns) + noise).clamp(-max_action, max_action)
```

**Summary table for the slide:**

| Problem in DDPG | TD3 fix | Config key |
|---|---|---|
| Q overestimation | Twin critics, use min | *(structural)* |
| Actor chases noisy critic | Delay actor updates | `policy_delay=2` |
| Sharp Q peaks at target action | Noise on target action | `policy_noise`, `noise_clip` |

**TD3Config extras over DDPGConfig:**  
[`benchmarks/td3.py` lines 26–30](../benchmarks/td3.py) — `policy_noise`, `noise_clip`, `policy_delay`.

---

### 35–50 min — SAC: principled exploration via entropy maximisation

**Slide focus:** the entropy-augmented objective and why it changes the whole policy design.

**Talking points (intro):**  
DDPG and TD3 explore through external noise added to a deterministic policy. SAC makes
exploration *intrinsic*: the policy is stochastic and is explicitly rewarded for having high
entropy. This changes the actor architecture from a deterministic mapping to a Gaussian
distribution.

**Concept doc:** [`03_sac.md`](03_sac.md)

#### 35–40 min — The entropy-regularised objective

**Talking points:**

Standard RL maximises cumulative reward:  
`J(π) = E[ Σ_t γᵗ·rₜ ]`

SAC augments this with the policy entropy `H(π(·|s))`, controlled by temperature `alpha`:  
`J_SAC(π) = E[ Σ_t γᵗ·(rₜ + α·H(π(·|sₜ))) ]`

Effect of `alpha`:
- High `alpha` → strong pressure to be stochastic → more exploration, slower convergence.
- Low `alpha` → mostly reward-driven → resembles TD3/DDPG.
- `alpha=0` → reduces to hard deterministic policy (similar to DDPG but with twin critics).

The entropy term then propagates into the **Bellman target** for the critic:  
`y = r + γ·(1-done)·[ min(Q1_target,Q2_target)(s',a') - α·log π(a'|s') ]`

And into the **actor loss**:  
`L_actor = E_{a~π}[ α·log π(a|s) - min(Q1,Q2)(s,a) ]`

#### 40–47 min — GaussianActor: stochastic action sampling with tanh squashing

**Talking points:**

- The actor outputs a **Gaussian distribution** `N(mu(s), sigma(s))`, not a single action.  
  Two output heads: `mean` and `log_std` (clamped to `[LOG_STD_MIN, LOG_STD_MAX] = [-20, 2]`).
- Action is drawn via reparameterisation: `z ~ N(mu, sigma)` then squashed:
  `action = tanh(z) * max_action`  
  Reparameterisation (`rsample`) makes the sample differentiable, so gradients flow through it.
- **Log-probability correction for tanh squashing:**  
  `log π(a|s) = log N(z|μ,σ) - Σᵢ log(1 - tanh²(zᵢ) + ε)`
  This is the change-of-variables formula. Without it `log_prob` would be incorrect and the
  entropy estimate would be wrong. The `1e-6` guards against `log(0)`.

**Code pointer — GaussianActor:**  
[`benchmarks/sac.py` lines 38–73](../benchmarks/sac.py)

```python
class GaussianActor(nn.Module):
    def forward(self, state):
        h = self.backbone(state)
        mean    = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self(state)
        std  = log_std.exp()
        z    = mean if deterministic else Normal(mean, std).rsample()
        squashed = torch.tanh(z)
        action   = squashed * self.max_action

        # tanh log-prob correction (change of variables)
        log_prob = Normal(mean, std).log_prob(z)
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = (log_prob - correction).sum(dim=1, keepdim=True)

        return action, log_prob
```

**Key point:** at **inference/video** time, `deterministic=True` gives `z = mean` and
`log_prob = 0`. The policy collapses to a deterministic mode, useful for evaluation.

#### 47–50 min — SAC update loop

**Talking points:**

- **No actor target network.** SAC does not need one: the stochastic policy naturally
  smooths the target surface (the role played by target-policy noise in TD3).
- **Critic target** still exists (slow-moving copy of the twin critic).
- Update order per step: critic → actor → critic target soft update.

**Code pointer — SAC update block:**  
[`benchmarks/sac.py` lines ~118–160](../benchmarks/sac.py)

```python
# Critic update — note entropy term (- alpha * next_logp) in target
next_a, next_logp = actor.sample(ns)
target_q = torch.min(*critic_target(ns, next_a)) - cfg.alpha * next_logp
target = r + nd * cfg.gamma * target_q
critic_loss = mse(critic(s,a)[0], target) + mse(critic(s,a)[1], target)

# Actor update — minimise (alpha*logp - min_Q)
pi_action, logp_pi = actor.sample(s)
min_q_pi = torch.min(*critic(s, pi_action))
actor_loss = (cfg.alpha * logp_pi - min_q_pi).mean()
```

**Structural difference vs DDPG/TD3 to emphasise on slide:**

| | DDPG | TD3 | SAC |
|---|---|---|---|
| Policy type | Deterministic | Deterministic | Stochastic Gaussian |
| Critics | 1 | 2 (min) | 2 (min) |
| Actor target | Yes | Yes | **No** |
| Exploration | External Gaussian noise | External Gaussian noise | **Intrinsic (entropy)** |
| Entropy in target | No | No | **Yes** |
| Extra hyperparameter | `exploration_noise` | `policy_noise`, `policy_delay` | `alpha` |

---

### 50–57 min — Side-by-side comparison and practical recommendations

**Slide focus:** run the benchmark, read the output table, make a decision.

#### Live demo (run during session if time allows, otherwise pre-recorded)

Run from `rl_continuous_optimization/`:

```bash
# Quick smoke test — all three algorithms, 160 episodes each
uv run python run_all_comparison.py \
    --methods ddpg td3 sac \
    --episodes 160 \
    --seed 42 \
    --output-dir outputs/session6
```

The script calls each algorithm sequentially, collects `BenchmarkResult` objects
(defined in [`benchmarks/common.py` lines 19–29](../benchmarks/common.py)), and prints a
summary table with:

- `max_avg_reward_100` — best 100-episode rolling average seen (peak performance)
- `final_avg_reward_100` — last 100 episodes (convergence stability)
- `elapsed_sec` — wall-clock time

**Expected output shape** (Pendulum-v1, seed 42):

```
Algorithm  | Max Avg Reward (100ep) | Final Avg Reward (100ep) | Time (s)
-----------|------------------------|--------------------------|--------
DDPG       | ~-200 to -150          | -200 to -160              | ~30s
TD3        | ~-150 to -100          | ~-140 to -110             | ~35s
SAC        | ~-120 to -80           | ~-120 to -90              | ~40s
```

> Pendulum-v1 reward range: –1600 (worst) to ~–120 (near-optimal).
> With only 160 episodes convergence is partial — this is intentional for demo speed.
> Production runs use 500–1000 episodes.

**Aggregate across seeds:**

```bash
# Run with multiple seeds
uv run python run_all_comparison.py --seed 0  --output-dir outputs/seed0
uv run python run_all_comparison.py --seed 1  --output-dir outputs/seed1
uv run python run_all_comparison.py --seed 42 --output-dir outputs/seed42

# Aggregate statistics
uv run python scripts/aggregate_results.py \
    --inputs outputs/seed*/comparison_results.json \
    --output-json outputs/aggregate_summary.json \
    --output-csv  outputs/aggregate_summary.csv

# Generate HTML report
uv run python scripts/generate_aggregate_report.py \
    --input  outputs/aggregate_summary.json \
    --output-dir outputs/report \
    --title "RL Continuous Control Aggregate Report"
```

Script [`scripts/aggregate_results.py`](../scripts/aggregate_results.py) computes `mean ± std`
per algorithm across seeds. This is the only statistically defensible way to compare RL
algorithms — single-seed numbers are unreliable.

#### Practical recommendations slide

| Scenario | Choice | Reason |
|---|---|---|
| First experiment on a new task | **SAC** | Robust, built-in exploration, little tuning |
| Need deterministic policy at deployment | **TD3** | Deterministic output, stable Q estimates |
| Teaching / ablation baseline | **DDPG** | Simplest code, all problems visible |
| Fastest wall-clock per sample | **DDPG** | Single critic, no log-prob computation |
| Best sample efficiency expected | **SAC or TD3** | Both significantly better than DDPG |

General rule: **start with SAC, compare with TD3, keep DDPG as the pedagogical anchor**.

---

### 57–60 min — Q&A guide

Anticipated questions and short answers:

**Q: Can we auto-tune `alpha` in SAC instead of fixing it?**  
A: Yes. Automatic entropy tuning treats `alpha` as a Lagrange multiplier targeting a desired
entropy level. It adds one Adam optimiser for `log_alpha`. Not implemented in this benchmark
(fixed `alpha=0.2`) but common in production SAC. Reference: Haarnoja et al. 2018 (v2).

**Q: Why is SAC slower per step than DDPG?**  
A: Each SAC update requires a `Normal.log_prob` computation and its gradient, plus the tanh
correction. This is cheap but not free. The trade-off is usually worth it.

**Q: Does SAC work for discrete action spaces?**  
A: The Gaussian actor is specifically for continuous spaces. Discrete SAC uses a
softmax policy and replaces reparameterisation with a direct expectation over all actions.
Not covered here.

**Q: When does exploration noise help more than entropy regularisation?**  
A: Exploration noise (DDPG/TD3) can be shaped (Ornstein-Uhlenbeck, schedule decay, domain
knowledge). Entropy regularisation (SAC) is more automatic but less customisable. For
highly constrained action spaces or safety-critical settings, explicit noise is often safer.

**Q: Is the replay buffer the same for all three algorithms?**  
A: Yes — see [`benchmarks/replay_buffer.py`](../benchmarks/replay_buffer.py). Same circular
buffer implementation with fixed `not_done` storage. The buffer is algorithm-agnostic.

---

## Complete concept-to-code map

| Concept | Algorithm doc | Code file | Key lines |
|---|---|---|---|
| Deterministic actor (tanh) | [`01_ddpg.md`](01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 35–48 |
| Single critic (state+action) | [`01_ddpg.md`](01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 51–63 |
| DDPG full train loop | [`01_ddpg.md`](01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 66–175 |
| Warmup + exploration noise | [`01_ddpg.md`](01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 101–108 |
| Critic + actor update + Polyak | [`01_ddpg.md`](01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 118–144 |
| Twin critic (Q1+Q2) | [`02_td3.md`](02_td3.md) | [`benchmarks/td3.py`](../benchmarks/td3.py) | 47–73 |
| Clipped double-Q target | [`02_td3.md`](02_td3.md) | [`benchmarks/td3.py`](../benchmarks/td3.py) | 119–121 |
| Target policy smoothing | [`02_td3.md`](02_td3.md) | [`benchmarks/td3.py`](../benchmarks/td3.py) | 113–118 |
| Delayed actor update | [`02_td3.md`](02_td3.md) | [`benchmarks/td3.py`](../benchmarks/td3.py) | 122–133 |
| GaussianActor (mean+log_std) | [`03_sac.md`](03_sac.md) | [`benchmarks/sac.py`](../benchmarks/sac.py) | 38–55 |
| reparameterisation + tanh squash | [`03_sac.md`](03_sac.md) | [`benchmarks/sac.py`](../benchmarks/sac.py) | 56–73 |
| SAC critic target (with entropy) | [`03_sac.md`](03_sac.md) | [`benchmarks/sac.py`](../benchmarks/sac.py) | 147–153 |
| SAC actor loss (entropy-regularised) | [`03_sac.md`](03_sac.md) | [`benchmarks/sac.py`](../benchmarks/sac.py) | 155–161 |
| Replay buffer | all three | [`benchmarks/replay_buffer.py`](../benchmarks/replay_buffer.py) | 1–43 |
| BenchmarkResult + metrics | all three | [`benchmarks/common.py`](../benchmarks/common.py) | 19–47 |
| Multi-method CLI runner | all three | [`run_all_comparison.py`](../run_all_comparison.py) | 1–end |
| Seed control | all three | [`benchmarks/common.py`](../benchmarks/common.py) | 31–34 |

---

## References

### DDPG
- Lillicrap et al. (2015) — original paper: https://arxiv.org/abs/1509.02971
- Explained: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-4643c1f71b2e/
- SpinningUp: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
- Coach implementation notes: https://intellabs.github.io/coach/components/agents/policy_optimization/ddpg.html
- Recent application: https://www.nature.com/articles/s41598-025-99213-3

### TD3
- Fujimoto et al. (2018) — original paper: https://arxiv.org/abs/1802.09477
- Explained: https://medium.com/@heyamit10/twin-delayed-deep-deterministic-policy-gradient-td3-fc8e9950f029
- SpinningUp: https://spinningup.openai.com/en/latest/algorithms/td3.html

### SAC
- Haarnoja et al. (2018) v1: https://arxiv.org/abs/1801.01290
- Haarnoja et al. (2018) v2 (auto-alpha): https://arxiv.org/abs/1812.05905
- Explained: https://towardsdatascience.com/navigating-soft-actor-critic-reinforcement-learning-8e1a7406ce48/
- SpinningUp: https://spinningup.openai.com/en/latest/algorithms/sac.html
