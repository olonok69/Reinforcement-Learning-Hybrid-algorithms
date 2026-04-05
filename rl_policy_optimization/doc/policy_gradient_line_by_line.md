# `benchmarks/policy_gradient.py` — Line-by-line explanation (English)

This companion document explains what each section of `benchmarks/policy_gradient.py` does and why it matters mathematically.

Main code file:
- [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)

Part 1 presentation guide:
- [presentation_guide_60min.md](presentation_guide_60min.md)

---

## 1) Imports and module purpose

- `from __future__ import annotations`
	- Enables postponed type-annotation evaluation; keeps hints more flexible.

- Standard imports (`dataclass`, `gymnasium`, `numpy`, `torch`, etc.)
	- `gymnasium`: environment API (`reset`, `step`)
	- `numpy`: reward aggregation
	- `torch`: neural network, autodiff, optimizer

- `from benchmarks.common import record_policy_video`
	- Shared utility to save evaluation rollouts as videos.

Relation to Part 1 theory:
- This file is the practical implementation of “Policy gradient theorem”, “Monte Carlo in REINFORCE”, and “REINFORCE algorithm” in [presentation_guide_60min.md](presentation_guide_60min.md).

---

## 2) `PolicyGradientConfig` (training controls)

`PolicyGradientConfig` centralizes hyperparameters:

- `env_name`
	- Gym task used for training (`CartPole-v1` by default).

- `gamma`
	- Discount factor from the Part 1 return definition:
	- $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$

- `learning_rate`
	- Adam step size.

- `episodes`
	- Number of Monte Carlo episodes for learning.

- `hidden_size`
	- Width of first hidden layer of the policy network.

- `normalize_returns`
	- Stabilization trick described in Monte Carlo section.

- `record_video`, `video_dir`, `video_episodes`
	- Optional post-training evaluation recording.

---

## 3) `PolicyNetwork` (state -> action logits)

### Constructor
- Builds an MLP with ReLU and final linear layer to number of actions.
- Output is **logits**, not probabilities.

### `forward`
- Runs input through the network and returns logits.

Relation to theory:
- In Part 1, policy is $\pi_\theta(a|s)$.
- In code, $\theta$ are all trainable parameters in this network.
- Probabilities are obtained later via `softmax(logits)`.

---

## 4) `_discounted_returns` (reward-to-go)

This helper computes $G_t$ for each timestep in an episode.

Mechanics:
1. Traverse rewards in reverse.
2. Keep accumulator: `running = r + gamma * running`.
3. Append each accumulated value.
4. Reverse list back to chronological order.

Why reverse iteration?
- Natural recursive return form:
- $G_t = r_t + \gamma G_{t+1}$

Part 1 connection:
- This is exactly the “reward-to-go” variant in policy-gradient variants section.

---

## 5) `run_policy_gradient` — complete pipeline

### 5.1 Setup
- `cfg = config or PolicyGradientConfig()`
	- Uses provided config or defaults.

- `env = gym.make(cfg.env_name)`
	- Creates environment instance.

- `obs_size`, `n_actions`
	- Reads observation/action dimensions from environment spaces.

- `net = PolicyNetwork(...)`
	- Instantiates policy model.

- `optimizer = optim.Adam(...)`
	- Optimizer that updates policy parameters.

### 5.2 Episode loop (Monte Carlo sampling)
- For each episode:
	- reset environment
	- rollout until terminal/truncated
	- collect:
		- `log_probs`: $\log \pi_\theta(a_t|s_t)$
		- `rewards`: reward sequence

Inside `while not done`:
- `state_t = torch.tensor(...).unsqueeze(0)`
	- Converts state to tensor and adds batch dimension.

- `logits = net(state_t)`
	- Forward pass.

- `probs = torch.softmax(logits, dim=1)`
	- Converts logits into a valid action distribution.

- `dist = torch.distributions.Categorical(probs)`
	- Stochastic policy distribution for discrete actions.

- `action = dist.sample()`
	- Samples action from policy (critical for unbiased estimator).

- `env.step(int(action.item()))`
	- Executes action and returns next state/reward.

- `done = terminated or truncated`
	- Handles Gymnasium stop conditions.

- `log_probs.append(dist.log_prob(action))`
	- Stores exact term required by policy-gradient theorem.

- `rewards.append(float(reward))`
	- Stores scalar reward for return computation.

### 5.3 REINFORCE loss construction
After episode ends:

- `returns = _discounted_returns(rewards, cfg.gamma)`
	- Computes reward-to-go $G_t$.

- `returns_t = torch.tensor(returns, dtype=torch.float32)`
	- Converts to tensor for vectorized operations.

- Optional normalization:
	- `(returns_t - mean) / (std + 1e-8)`
	- Reduces gradient-scale volatility.

- `policy_loss = -torch.stack(log_probs) * returns_t`
	- Per-step REINFORCE term:
	- $-\log\pi_\theta(a_t|s_t) G_t$

- `loss = policy_loss.sum()`
	- Sum over episode timesteps.

Why minus sign?
- Optimizer performs gradient **descent** on `loss`.
- We want gradient **ascent** on expected return.
- Negation makes them equivalent.

### 5.4 Backprop and update
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

This computes and applies parameter updates.

### 5.5 Reward tracking
- `ep_reward = np.sum(rewards)`
- `episode_rewards.append(ep_reward)`

Used later for benchmark metrics (`max_avg_reward_100`, `final_avg_reward_100`).

### 5.6 Optional video recording
If `cfg.record_video`:
- `net.eval()` sets evaluation mode.
- Defines deterministic `_policy` with `argmax(logits)`.
- Calls `record_policy_video(...)` helper.

Important distinction:
- Training uses stochastic sampled actions.
- Evaluation video uses deterministic greedy policy for reproducibility.

### 5.7 Cleanup and return
- `env.close()` frees environment resources.
- `return episode_rewards` returns reward history to benchmark runner.

---

## 6) Direct map to Part 1 explanations

Use these links while presenting:

- Objective and gradient ascent:
	- [presentation_guide_60min.md](presentation_guide_60min.md)
	- Sections: “Policy optimization from first principles”, “Policy gradient theorem”.

- Monte Carlo and reward-to-go:
	- [presentation_guide_60min.md](presentation_guide_60min.md)
	- Sections: “Monte Carlo in REINFORCE”, “Policy gradient variants”.

- Loss expression and optimizer step:
	- [presentation_guide_60min.md](presentation_guide_60min.md)
	- Section: “REINFORCE algorithm (step-by-step)”.

- Code explained:
	- [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
