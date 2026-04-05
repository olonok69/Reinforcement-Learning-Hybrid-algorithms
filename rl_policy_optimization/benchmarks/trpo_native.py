from __future__ import annotations
"""Native TRPO implementation for discrete-action Gymnasium tasks.

This module provides an in-repo TRPO trainer (no sb3 dependency) so we can:
- inspect and teach trust-region internals directly,
- run side-by-side comparisons against sb3 backend,
- keep the same benchmark output contract (`list[float]` rewards).
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from benchmarks.common import record_policy_video
from benchmarks.trpo_core import backtracking_line_search, conjugate_gradients, flat_grad, flat_params, set_flat_params


@dataclass
class TRPONativeConfig:
    """Hyperparameters and runtime options for native TRPO training."""

    # Environment identifier used by Gymnasium.
    env_name: str = "CartPole-v1"
    # Global training budget in environment timesteps.
    total_timesteps: int = 100_000
    # Number of transitions collected before each TRPO update.
    batch_size: int = 2_048
    # Discount factor for future rewards.
    gamma: float = 0.99
    # GAE smoothing coefficient (bias/variance trade-off).
    gae_lambda: float = 0.97
    # Trust-region target: upper bound for average KL change.
    max_kl: float = 1e-2
    # Numerical damping for Fisher-vector product stabilization.
    damping: float = 1e-1
    # Number of conjugate-gradient iterations.
    cg_steps: int = 10
    # Critic optimizer learning rate.
    value_lr: float = 1e-3
    # Number of critic updates per collected batch.
    value_iters: int = 40
    # Hidden width for both policy and value MLPs.
    hidden_size: int = 128
    # Optional post-training video recording flags.
    record_video: bool = False
    video_dir: str = "videos/trpo_native"
    video_episodes: int = 3


class CategoricalPolicy(nn.Module):
    """Discrete-action policy network.

    Outputs action logits for a categorical distribution π(a|s).
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        # Register layers/parameters with PyTorch module base class.
        super().__init__()
        # Shared MLP trunk ending in action logits.
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # Return raw logits; caller constructs Categorical distribution.
        return self.net(states)


class ValueNet(nn.Module):
    """State-value approximator V(s) used for advantage estimation."""

    def __init__(self, obs_size: int, hidden_size: int = 128):
        # Register layers/parameters with PyTorch module base class.
        super().__init__()
        # MLP that predicts scalar value for each state.
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # Squeeze trailing singleton dimension to return shape [batch].
        return self.net(states).squeeze(-1)


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and bootstrapped returns for one rollout.

    This mirrors standard PPO/TRPO-style GAE recursion:
      δ_t = r_t + γ V(s_{t+1}) - V(s_t)
      A_t = δ_t + γ λ A_{t+1}
    """
    # Same length as rewards rollout.
    advantages = np.zeros_like(rewards, dtype=np.float32)
    # Backward-running GAE accumulator.
    gae = 0.0
    # Iterate backward so each step can use already-computed next advantage.
    for t in reversed(range(len(rewards))):
        # Disable bootstrap term when current transition ends an episode.
        not_done = 1.0 - float(dones[t])
        # Use provided bootstrap for final element; otherwise next stored value.
        next_v = next_value if t == len(rewards) - 1 else values[t + 1]
        # One-step TD residual.
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        # GAE recursion.
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
    # Critic target: return = advantage + baseline value.
    returns = advantages + values
    return advantages, returns


def run_trpo_native(config: TRPONativeConfig | None = None) -> list[float]:
    """Train native TRPO and return per-episode rewards.

    The training loop alternates between:
    1) rollout collection,
    2) critic fitting,
    3) trust-region actor update (CG + line search).
    """
    # Resolve runtime config.
    cfg = config or TRPONativeConfig()
    print("\n--- Starting TRPO (Native) ---")

    # Build environment and infer observation/action dimensions.
    env = gym.make(cfg.env_name)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create policy (actor) and value function (critic).
    policy = CategoricalPolicy(obs_size, n_actions, hidden_size=cfg.hidden_size)
    value_net = ValueNet(obs_size, hidden_size=cfg.hidden_size)
    # Critic is optimized with standard first-order optimizer.
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    # Benchmark reward history: one scalar per completed episode.
    rewards_history: list[float] = []
    # Global timestep counter against training budget.
    timestep_count = 0
    # Initialize environment state.
    state, _ = env.reset()
    # Running reward accumulator for current episode.
    episode_reward = 0.0

    # Outer training loop: continue until timesteps budget is exhausted.
    while timestep_count < cfg.total_timesteps:
        # Rollout buffers for one TRPO batch.
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []
        old_log_probs: list[float] = []
        values: list[float] = []

        # Collect at most `batch_size` transitions for this policy update.
        collected = 0
        while collected < cfg.batch_size and timestep_count < cfg.total_timesteps:
            # Convert numpy state to batched float tensor.
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # Rollout sampling and value evaluation run without grad tracking.
            with torch.no_grad():
                # Policy forward pass -> logits -> categorical distribution.
                logits = policy(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                # Stochastic action sampling (required for policy-gradient estimate).
                action_t = dist.sample()
                # Cache old log-prob for importance-ratio term in surrogate loss.
                log_prob_t = dist.log_prob(action_t)
                # Critic baseline estimate for current state.
                value_t = value_net(state_t)

            # Step environment with sampled action.
            action = int(action_t.item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store rollout transition pieces.
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            dones.append(done)
            old_log_probs.append(float(log_prob_t.item()))
            values.append(float(value_t.item()))

            # Update running episode and rollout counters.
            episode_reward += float(reward)
            state = next_state
            collected += 1
            timestep_count += 1

            if done:
                # Persist completed episode reward for benchmark metrics.
                rewards_history.append(episode_reward)
                episode_reward = 0.0
                # Start next episode immediately within current rollout collection.
                state, _ = env.reset()

        # Safety guard for degenerate empty rollout.
        if not states:
            continue

        # Bootstrap value at state right after last collected transition.
        with torch.no_grad():
            next_value = float(value_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item())

        # Convert Python lists into contiguous NumPy arrays.
        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int64)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.bool_)
        values_np = np.asarray(values, dtype=np.float32)
        old_log_probs_np = np.asarray(old_log_probs, dtype=np.float32)

        # Compute advantages/returns for actor and critic training.
        adv_np, returns_np = _compute_gae(
            rewards_np,
            dones_np,
            values_np,
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        # Normalize advantages for more stable policy optimization.
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        # Convert batch arrays to torch tensors.
        states_t = torch.tensor(states_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.int64)
        old_log_probs_t = torch.tensor(old_log_probs_np, dtype=torch.float32)
        advantages_t = torch.tensor(adv_np, dtype=torch.float32)
        returns_t = torch.tensor(returns_np, dtype=torch.float32)

        # Critic phase: fit V(s) to bootstrapped returns.
        for _ in range(cfg.value_iters):
            pred_values = value_net(states_t)
            value_loss = nn.functional.mse_loss(pred_values, returns_t)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        # Snapshot old policy probabilities for KL and importance ratios.
        with torch.no_grad():
            old_logits = policy(states_t)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            old_probs = old_dist.probs.detach()

        # Freeze parameter tuple once for repeated autograd calls.
        params = tuple(policy.parameters())

        def surrogate_loss() -> torch.Tensor:
            # Compute new policy log-probabilities on stored actions.
            logits = policy(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            # Importance ratio π_new / π_old.
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            # Negative sign because optim step is framed as minimization.
            return -(ratio * advantages_t).mean()

        def mean_kl() -> torch.Tensor:
            # KL(π_old || π_new) averaged over batch states.
            logits = policy(states_t)
            logp_new = torch.log_softmax(logits, dim=1)
            logp_old = torch.log(old_probs + 1e-8)
            kl = (old_probs * (logp_old - logp_new)).sum(dim=1)
            return kl.mean()

        # Gradient of surrogate objective wrt policy parameters.
        loss = surrogate_loss()
        grads = torch.autograd.grad(loss, params)
        loss_grad = flat_grad(grads, params).detach()

        def fisher_vector_product(vec: torch.Tensor) -> torch.Tensor:
            # Build Hessian-vector product of KL term via autograd trick.
            kl = mean_kl()
            kl_grads = torch.autograd.grad(kl, params, create_graph=True)
            flat_kl_grad = flat_grad(kl_grads, params)
            # Directional derivative of KL gradient along `vec`.
            kl_v = (flat_kl_grad * vec).sum()
            hvp_grads = torch.autograd.grad(kl_v, params)
            hvp = flat_grad(hvp_grads, params).detach()
            # Damped Fisher-vector product for numerical stability.
            return hvp + cfg.damping * vec

        # Solve F x = -g approximately for natural-gradient step direction.
        step_dir = conjugate_gradients(fisher_vector_product, -loss_grad, cfg.cg_steps)
        # Quadratic model term used to scale step to satisfy KL budget.
        shs = 0.5 * (step_dir * fisher_vector_product(step_dir)).sum()
        # Lagrange multiplier approximation.
        lm = torch.sqrt(torch.clamp(shs / cfg.max_kl, min=1e-8))
        # Full trust-region step candidate.
        full_step = step_dir / (lm + 1e-8)

        # Keep previous parameters so we can rollback if needed.
        prev = flat_params(policy).detach().clone()

        def evaluate_current() -> tuple[torch.Tensor, torch.Tensor]:
            # Evaluate objective + KL at current temporary parameter values.
            new_loss = surrogate_loss().detach()
            new_kl = mean_kl().detach()
            return new_loss, new_kl

        # Conservative candidate selection via backtracking.
        success, candidate = backtracking_line_search(
            policy=policy,
            prev_params=prev,
            full_step=full_step,
            evaluate=evaluate_current,
        )

        # Commit accepted parameters, then enforce hard KL safety check.
        with torch.no_grad():
            if success:
                set_flat_params(policy, candidate)
            # If KL still violates budget, revert to previous safe params.
            if mean_kl().item() > cfg.max_kl:
                set_flat_params(policy, prev)

    # Optional deterministic video recording after training.
    if cfg.record_video:
        # Switch to eval mode for deterministic inference behavior.
        policy.eval()

        # Greedy policy for clearer demos (argmax instead of sampling).
        def _policy(state_np: np.ndarray) -> int:
            with torch.no_grad():
                state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
                logits = policy(state_t)
                return int(torch.argmax(logits, dim=1).item())

            # Shared helper handles RecordVideo wrapper and episode capture.
        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="trpo_native",
            policy_fn=_policy,
        )

    # Explicit environment cleanup.
    env.close()
    # Return episodic rewards for downstream benchmark summaries.
    return rewards_history
