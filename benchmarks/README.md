# Benchmarks — Code Guide

This package contains the self-contained implementations of DDPG, TD3, and SAC used in the
RL Continuous Control session. Every algorithm follows the same structural contract so you can
read one and immediately orient yourself in the others.

**Presenter guides:**  
[`../doc/presentation_guide_60min.md`](../doc/presentation_guide_60min.md) (EN) ·
[`../doc/presentation_guide_60min_es.md`](../doc/presentation_guide_60min_es.md) (ES)

---

## Package structure

```
benchmarks/
├── __init__.py        ← public re-exports
├── common.py          ← BenchmarkResult, metrics, seed control, video helper
├── replay_buffer.py   ← shared off-policy experience store (used by all three)
├── ddpg.py            ← DDPG: DDPGConfig · Actor · Critic · run_ddpg()
├── td3.py             ← TD3:  TD3Config  · Actor · TwinCritic · run_td3()
└── sac.py             ← SAC:  SACConfig  · GaussianActor · TwinCritic · run_sac()
```

Every algorithm file follows the same pattern:

```
1. Config dataclass      (hyperparameters with safe defaults)
2. Network class(es)     (actor, critic)
3. run_<algo>() function (env setup → train loop → optional video → return reward history)
```

---

## `replay_buffer.py` — The shared experience store

All three algorithms are off-policy and share this exact same buffer.

### What it does

Stores transitions `(s, a, s', r, done)` in pre-allocated NumPy arrays using a circular
pointer. When full, the oldest transitions are overwritten. Sampling is uniform random, which
breaks temporal correlation and allows SGD to work correctly.

### Key implementation details

```python
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        # Pre-allocate fixed-size arrays — no dynamic allocation during training
        self.state      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.action     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.reward     = np.zeros((max_size, 1),          dtype=np.float32)
        self.not_done   = np.zeros((max_size, 1),          dtype=np.float32)
        # ptr  = write cursor (wraps at max_size)
        # size = number of valid entries (saturates at max_size)
```

**`not_done` instead of `done`** — stores `1.0 - done`. This allows the Bellman target to be
written cleanly as `r + not_done * gamma * target_q` without a conditional inside the vectorised
update. All three algorithm files rely on this convention.

```python
def add(self, state, action, next_state, reward, done):
    self.not_done[self.ptr] = 1.0 - float(done)  # ← stored inverted
    self.ptr  = (self.ptr + 1) % self.max_size    # ← circular wrap
    self.size = min(self.size + 1, self.max_size)  # ← saturate at capacity

def sample(self, batch_size):
    idx = np.random.randint(0, self.size, size=batch_size)  # uniform random
    # returns 5 tensors already on the configured device
    return (
        torch.tensor(self.state[idx],      device=self.device),
        torch.tensor(self.action[idx],     device=self.device),
        torch.tensor(self.next_state[idx], device=self.device),
        torch.tensor(self.reward[idx],     device=self.device),
        torch.tensor(self.not_done[idx],   device=self.device),
    )
```

**Presenter note:** point out that `sample()` is the only place where NumPy → PyTorch conversion
happens. Inside the training loop everything is already on `device`.

**Default capacity:** `buffer_size=200_000` in all three configs (≈ 1000 episodes of 200 steps).
This is generous for Pendulum-v1 but small for harder environments.

---

## `common.py` — Shared utilities

### `BenchmarkResult` (lines 19–29)

```python
@dataclass
class BenchmarkResult:
    algo: str
    episodes: int
    elapsed_sec: float
    max_avg_reward_100: float    # best 100-ep rolling mean (peak performance)
    final_avg_reward_100: float  # last 100 episodes (convergence stability)
```

`max_avg_reward_100` rewards algorithms that converge early and stay there.  
`final_avg_reward_100` rewards algorithms that are still improving at the end.  
Both together give a fuller picture than a single final reward.

### `set_global_seed` (lines 31–34)

```python
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

Seeds `random`, `numpy`, and `torch` simultaneously. Called once from each CLI entry point
before any environment or network is created. **Does not seed the Gymnasium environment** —
pass `seed` to `env.reset(seed=seed)` if you need full reproducibility down to env dynamics.

### `run_timed` (lines 37–49)

Wraps a training function, measures wall-clock time, and returns a `BenchmarkResult`:

```python
def run_timed(train_fn, algo) -> tuple[list[float], BenchmarkResult]:
    start   = time.time()
    rewards = list(train_fn())
    elapsed = time.time() - start
    result  = BenchmarkResult(
        algo=algo,
        episodes=len(rewards),
        elapsed_sec=elapsed,
        max_avg_reward_100  = moving_average_max(rewards,  window=100),
        final_avg_reward_100= moving_average_last(rewards, window=100),
    )
    return rewards, result
```

### `moving_average_max` / `moving_average_last` (lines 39–47)

Both use `np.convolve` with a uniform kernel for efficiency. `moving_average_max` slides a
window across the full history and returns the maximum — useful for detecting whether an
algorithm ever reached a good region even if it later destabilised.

### `record_policy_video_continuous` (lines 79–119)

Wraps the trained policy in Gymnasium's `RecordVideo` wrapper and runs `episodes` evaluation
episodes. Uses `deterministic=True` mode for SAC (mean action, no sampling noise). The function
is optional — enable with `--record-video` CLI flag.

---

## `ddpg.py` — DDPG implementation

**Concept doc:** [`../doc/01_ddpg.md`](../doc/01_ddpg.md) (EN) · [`../doc/es/01_ddpg.md`](../doc/es/01_ddpg.md) (ES)

### `DDPGConfig` (lines 15–32)

```python
@dataclass
class DDPGConfig:
    env_name:               str   = "Pendulum-v1"
    episodes:               int   = 160
    max_steps:              int   = 200     # max steps per episode (Pendulum cap)
    gamma:                  float = 0.99    # discount factor
    tau:                    float = 0.005   # Polyak soft-update rate
    actor_lr:               float = 1e-3   # Adam lr for actor
    critic_lr:              float = 1e-3   # Adam lr for critic
    hidden_size:            int   = 256    # neurons per hidden layer (both nets)
    batch_size:             int   = 256    # mini-batch size for each update
    buffer_size:            int   = 200_000
    warmup_steps:           int   = 5_000  # random-action steps before first update
    exploration_noise:      float = 0.1    # Gaussian noise std (× max_action)
    train_updates_per_step: int   = 1      # SGD updates per env step
```

`tau=0.005` is the canonical value from the DDPG paper. Larger values track the online
network faster but reduce target stability. `exploration_noise=0.1` means 10% of the action
range as 1-sigma noise — mild but sufficient for Pendulum-v1.

### `Actor` (lines 35–48)

```python
class Actor(nn.Module):
    # Architecture: Linear(state_dim→hidden) → ReLU → Linear → ReLU → Linear(→action_dim)
    # Output: tanh(net(state)) * max_action
```

Two key design choices:
1. **`tanh` output activation** — squashes pre-activations to `(-1, 1)`, then rescales to
   `(-max_action, max_action)`. This guarantees the output is always within the action bounds
   without any explicit clipping inside the forward pass.
2. **`max_action` scaling** — stored as an attribute to handle environments where
   `action_space.high != 1.0`.

### `Critic` (lines 51–63)

```python
class Critic(nn.Module):
    # Architecture: Linear(state_dim+action_dim→hidden) → ... → Linear(→1)
    # Input: torch.cat([state, action], dim=1)
```

The critic takes the **concatenation of state and action** as a single vector input. This
is the standard approach for DDPG-family critics because it allows the gradient
`dQ/da → d(actor)/d(theta_actor)` to flow cleanly during the actor update.

### `run_ddpg()` — training loop walkthrough

#### Network instantiation + target copy (lines 77–83)
```python
actor_target.load_state_dict(actor.state_dict())   # identical initialisation
critic_target.load_state_dict(critic.state_dict())
```
Both targets start as exact copies. They are never trained directly — they only receive
Polyak updates.

#### Warmup phase (lines 101–102)
```python
if global_step < cfg.warmup_steps:
    action = env.action_space.sample()  # pure random, fills buffer with diverse data
```
The first 5000 env steps take random actions regardless of the episode number. This ensures
the buffer has diverse experience before the first gradient step — critical for off-policy
stability.

#### Exploration during training (lines 103–108)
```python
action = actor(state_t).cpu().numpy()
noise  = np.random.normal(0.0, cfg.exploration_noise * max_action, size=action_dim)
action = np.clip(action + noise, env.action_space.low, env.action_space.high)
```
Exploration is extrinsic Gaussian noise added to the deterministic action.  
`np.clip` enforces hard action bounds after noise addition — avoids out-of-range actions
passed to the environment.

#### Guard condition (line 118)
```python
if replay.size >= max(cfg.warmup_steps, cfg.batch_size):
```
Training starts only when there are enough transitions to fill a batch *and* the warmup
period has elapsed. `max(...)` handles edge cases where `batch_size > warmup_steps`.

#### Critic update (lines 122–132)
```python
# Step 1: Bellman target — computed under torch.no_grad() (no target gradients)
next_action = actor_target(ns)
target_q    = critic_target(ns, next_action)
target_q    = r + nd * cfg.gamma * target_q   # nd = not_done = (1 - done)

# Step 2: MSE between online Q and target Q
critic_loss = F.mse_loss(critic(s, a), target_q)
critic_opt.zero_grad()
critic_loss.backward()
critic_opt.step()
```
`torch.no_grad()` around target computations is essential — we must not backprop through
the target networks.

#### Actor update (lines 134–138)
```python
actor_loss = -critic(s, actor(s)).mean()  # maximise Q → minimise -Q
actor_opt.zero_grad()
actor_loss.backward()
actor_opt.step()
```
The actor gradient chain: `actor_loss → critic(s, actor(s)) → actor(s) → actor.parameters()`.  
The critic is used as a **differentiable reward signal** for the actor. The critic's own
weights are not updated here (no `critic_opt.step()`).

#### Soft (Polyak) update (lines 140–144)
```python
for p, tp in zip(critic.parameters(), critic_target.parameters()):
    tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)  # tp = (1-tau)*tp + tau*p
for p, tp in zip(actor.parameters(), actor_target.parameters()):
    tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)
```
Done *in-place* with `mul_` and `add_` to avoid creating new tensors. With `tau=0.005` the
target moves at 0.5% of the online weight per step — a very slow tracking that provides
stable Bellman targets.

---

## `td3.py` — TD3 implementation

**Concept doc:** [`../doc/02_td3.md`](../doc/02_td3.md) (EN) · [`../doc/es/02_td3.md`](../doc/es/02_td3.md) (ES)

TD3 shares the same `Actor` architecture as DDPG. All differences are in `TD3Config`,
`TwinCritic`, and the update block. The diff against DDPG is the best way to explain it.

### `TD3Config` — new fields vs `DDPGConfig` (lines 26–30)

```python
policy_noise:  float = 0.2   # std of noise added to target actor (Fix 3)
noise_clip:    float = 0.5   # max absolute value of that noise (Fix 3)
policy_delay:  int   = 2     # critic updates per actor update (Fix 2)
```

These three fields are the direct implementation of TD3's three fixes. `exploration_noise`
is still present for the online exploration phase (same role as in DDPG).

### `TwinCritic` (lines 47–73)

```python
class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        self.q1 = nn.Sequential(...)  # independent MLP — same architecture as DDPG Critic
        self.q2 = nn.Sequential(...)  # independent MLP — different random init → different estimates

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)  # returns both Q-values

    def q1_only(self, state, action):    # used for actor policy gradient (only Q1)
        return self.q1(torch.cat([state, action], dim=1))
```

`q1_only` exists because the actor is updated using only `Q1` (to avoid the actor exploiting the
minimum — the actor should push toward good states, not chase the pessimistic lower bound).

### Update block — three fixes in context

#### Fix 3 — Target policy smoothing (lines 113–121)
```python
noise = (torch.randn_like(a) * cfg.policy_noise * max_action).clamp(
    -cfg.noise_clip * max_action,
     cfg.noise_clip * max_action,
)
next_action = (actor_target(ns) + noise).clamp(-max_action, max_action)
```
Noise is generated from the stored action tensor shape (`randn_like`), scaled, clipped at
`noise_clip`, applied to the target action, then the result is clamped to action bounds.
The double-clip (inner `clamp` on noise + outer `clamp` on action) prevents extreme targets.

#### Fix 1 — Clipped double-Q (lines 119–121)
```python
target_q1, target_q2 = critic_target(ns, next_action)
target_q = r + nd * cfg.gamma * torch.min(target_q1, target_q2)  # pessimistic target
```
`torch.min(target_q1, target_q2)` is element-wise — each sample in the batch gets the lower of
its two Q estimates. This counters overestimation by never bootstrapping from an optimistic critic.

#### Critic loss — both heads trained toward the same target (lines 123–128)
```python
current_q1, current_q2 = critic(s, a)
critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
```
Both critics train toward the same `target_q`. Over time they will diverge slightly due to
different random initialisations and different gradient histories, which is what creates the
useful diversity between `Q1` and `Q2`.

#### Fix 2 — Delayed actor + target update (lines 130–139)
```python
if update_step % cfg.policy_delay == 0:
    actor_loss = -critic.q1_only(s, actor(s)).mean()
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

    # Soft update both targets — ONLY when actor updates
    with torch.no_grad():
        for p, tp in zip(critic.parameters(), critic_target.parameters()):
            tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)
        for p, tp in zip(actor.parameters(), actor_target.parameters()):
            tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)
```
The target soft update is gated inside the `if` block — targets only move when the actor moves.
This keeps the actor-target and critic-target advances synchronised with each other.

---

## `sac.py` — SAC implementation

**Concept doc:** [`../doc/03_sac.md`](../doc/03_sac.md) (EN) · [`../doc/es/03_sac.md`](../doc/es/03_sac.md) (ES)

SAC has the largest structural divergence from DDPG/TD3 because the policy is now a probability
distribution, not a deterministic function.

### Constants (lines 17–18)

```python
LOG_STD_MIN = -20.0
LOG_STD_MAX =  2.0
```

These bound the log-standard-deviation output of the Gaussian actor:
- `LOG_STD_MAX = 2.0` → max `std = e^2 ≈ 7.4` — allows highly exploratory behaviour
- `LOG_STD_MIN = -20.0` → min `std = e^{-20} ≈ 2×10^{-9}` — near-deterministic

Without clamping, the network can push `log_std` to ±∞, causing NaN in the normal distribution.

### `SACConfig` — key difference vs DDPG/TD3 (lines 21–35)

```python
alpha: float = 0.2  # entropy temperature — controls exploration-exploitation tradeoff
# No exploration_noise — SAC never needs it; the stochastic policy handles exploration
```

Notice `exploration_noise` is absent from `SACConfig`. During the warmup phase SAC uses
`env.action_space.sample()`, and during training `actor.sample(s, deterministic=False)`
provides stochastic actions intrinsically.

### `GaussianActor` (lines 38–73)

This is the most important class to understand, containing SAC's defining mechanisms.

#### Architecture (lines 43–56)
```python
class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, max_action):
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mean    = nn.Linear(hidden_size, action_dim)  # mean of Gaussian
        self.log_std = nn.Linear(hidden_size, action_dim)  # log-std of Gaussian
        self.max_action = max_action
```

Shared backbone → two parallel linear heads. This is more parameter-efficient than two
separate networks and ensures consistent representation for both outputs.

#### `sample()` — the reparameterisation trick + tanh squashing (lines 58–73)
```python
def sample(self, state, deterministic=False):
    mean, log_std = self(state)
    std = log_std.exp()

    # Reparameterisation: z = mean + std * epsilon, epsilon ~ N(0,1)
    # rsample() implements this; the sample is differentiable w.r.t. mean and std
    z = mean if deterministic else Normal(mean, std).rsample()

    squashed = torch.tanh(z)         # squash to (-1, 1)
    action   = squashed * self.max_action  # rescale to action bounds

    if deterministic:
        log_prob = torch.zeros((state.shape[0], 1), ...)  # unused in eval
    else:
        # Log-probability of z under the pre-squash Gaussian
        log_prob  = Normal(mean, std).log_prob(z)
        # Change-of-variables correction for the tanh transformation:
        # d(tanh(z))/dz = 1 - tanh²(z)  →  subtract log of this Jacobian
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)  # 1e-6 prevents log(0)
        # Sum over action dimensions (independent per-dimension Gaussians)
        log_prob = (log_prob - correction).sum(dim=1, keepdim=True)

    return action, log_prob
```

**Why `rsample` and not `sample`?**  
`sample()` uses the standard `N(0,1)` trick but the result is not differentiable w.r.t.
the distribution parameters. `rsample()` (reparameterised sample) expresses the sample as
`mean + std * z`, where `z ~ N(0,1)` is treated as a fixed constant during backprop,
making gradients flow through `mean` and `std` to the network weights.

**The tanh-squash correction is non-trivial and often misunderstood.** If we ignored it and
used the raw Gaussian log-prob, the entropy estimate would be wrong because the distribution
after `tanh` is no longer Gaussian — its density is distorted. The correction term is the log
of the Jacobian determinant of the inverse transform.

**`deterministic=True`** collapses the policy to its mean action with zero log-prob. Used at
evaluation time (video recording, policy comparison) to get a single deterministic output.

### `TwinCritic` in SAC (lines 76–95)

Identical architecture to TD3's `TwinCritic` but without `q1_only` (SAC always uses the minimum
of both for both the target and the actor update). The actor update uses:
```python
min_q_pi = torch.min(q1_pi, q2_pi)
actor_loss = (cfg.alpha * logp_pi - min_q_pi).mean()
```
Both heads are queried and the minimum is taken — unlike TD3 where only `Q1` drives the actor.

### `run_sac()` — key differences from DDPG/TD3

#### No actor target network
SAC creates `critic_target` but **not** `actor_target`. The stochastic policy naturally
regularises the Q surface (the same goal as TD3's target-policy smoothing).

#### Critic target includes entropy (lines ~147–153)
```python
with torch.no_grad():
    next_a, next_logp = actor.sample(ns, deterministic=False)
    target_q1, target_q2 = critic_target(ns, next_a)
    target_q = torch.min(target_q1, target_q2) - cfg.alpha * next_logp
    target    = r + nd * cfg.gamma * target_q
```
`- cfg.alpha * next_logp` subtracts the entropy cost from the target. High-entropy actions
(negative `log_prob`) increase the target value — the critic learns to assign higher Q to
states where the policy has high entropy, which in turn encourages the actor to maintain it.

#### Actor loss (lines ~155–161)
```python
pi_action, logp_pi = actor.sample(s, deterministic=False)
q1_pi, q2_pi = critic(s, pi_action)
min_q_pi  = torch.min(q1_pi, q2_pi)
actor_loss = (cfg.alpha * logp_pi - min_q_pi).mean()
```
The actor minimises `alpha * log_pi - min_Q`:
- `- min_Q` → move toward high-value actions (same as DDPG actor loss direction)
- `+ alpha * log_pi` → penalise low-entropy actions (keep the policy spread out)

The balance between these two terms is entirely controlled by `alpha`.

#### Soft update — critic only (no actor to update targets for)
```python
with torch.no_grad():
    for p, tp in zip(critic.parameters(), critic_target.parameters()):
        tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)
# No actor target update
```

---

## Algorithm comparison at a glance

| Aspect | DDPG | TD3 | SAC |
|---|---|---|---|
| **Policy type** | Deterministic | Deterministic | Stochastic Gaussian |
| **Actor class** | `Actor` (`tanh`) | `Actor` (`tanh`) | `GaussianActor` (mean + log_std) |
| **Critic class** | `Critic` (×1) | `TwinCritic` (×2) | `TwinCritic` (×2) |
| **Target networks** | Actor + Critic | Actor + Critic | **Critic only** |
| **Bellman target** | Single Q | `min(Q1,Q2)` | `min(Q1,Q2) - α·log_π` |
| **Actor update** | Every step | Every `policy_delay` steps | Every step |
| **Actor uses** | `Q(s, μ(s))` | `Q1(s, μ(s))` | `min(Q1,Q2) - α·log_π` |
| **Exploration** | Additive Gaussian | Additive Gaussian | Stochastic sampling |
| **Actor target noise** | None | `policy_noise` + `noise_clip` | N/A |
| **Lines of unique logic** | ~35 | ~50 (+3 config fields) | ~65 (+`alpha`) |

---

## Running the benchmarks

### Individual algorithm

```bash
# From rl_continuous_optimization/
uv run python ddpg_benchmark.py --episodes 160 --seed 42
uv run python td3_benchmark.py  --episodes 160 --seed 42
uv run python sac_benchmark.py  --episodes 160 --seed 42
```

All three CLI runners accept the same core arguments plus algorithm-specific ones.
See `--help` for the full list. Config fields in each `*Config` dataclass map 1:1 to CLI flags.

### Unified comparison

```bash
uv run python run_all_comparison.py \
    --methods ddpg td3 sac \
    --seed 42 \
    --output-dir outputs/session6
```

Output: `outputs/session6/comparison_results.json` and `comparison_results.csv`.

### Multi-seed statistical comparison

```bash
for seed in 0 1 2 42; do
  uv run python run_all_comparison.py --seed $seed --output-dir outputs/seed${seed}
done

uv run python scripts/aggregate_results.py \
    --inputs outputs/seed*/comparison_results.json \
    --output-json outputs/aggregate_summary.json \
    --output-csv  outputs/aggregate_summary.csv

uv run python scripts/generate_aggregate_report.py \
    --input  outputs/aggregate_summary.json \
    --output-dir outputs/report \
    --title "RL Continuous Control Aggregate Report"
```

### Video recording

Any benchmark can record evaluation episodes after training:

```bash
uv run python sac_benchmark.py --record-video --video-dir videos/sac --video-episodes 3
```

Requires `moviepy`: `uv pip install moviepy`.

---

## Common debugging checklist

| Symptom | Likely cause | Where to look |
|---|---|---|
| Rewards stay near –1600 (worst) | Warmup too long, buffer too small, LR too high | `warmup_steps`, `buffer_size`, `actor_lr`/`critic_lr` in Config |
| NaN loss in SAC | `log_std` unclamped or `log(0)` in correction | `LOG_STD_MIN/MAX` constants, `1e-6` in correction |
| DDPG diverges quickly | Q-overestimation runaway with single critic | Switch to TD3; reduce `actor_lr`; reduce `tau` |
| TD3 very slow to improve | `policy_delay` too large; target noise too high | `policy_delay=2`, `policy_noise=0.1`; check `noise_clip` |
| SAC ignores reward | `alpha` too high | Reduce `alpha` from 0.2 to 0.05–0.1; consider auto-alpha |
| Results differ across runs | Seed not set | Call `set_global_seed(seed)` before env and network creation |
