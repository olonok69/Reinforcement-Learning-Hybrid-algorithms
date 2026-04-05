from __future__ import annotations
"""Proximal Policy Optimization (PPO) benchmark on CartPole-v1."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.common import record_policy_video


@dataclass
class PPOConfig:
    """Configuration for PPO training and update schedule."""

    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    episodes: int = 500
    hidden_size: int = 128
    rollout_steps: int = 1024
    update_epochs: int = 4
    minibatch_size: int = 64
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    record_video: bool = False
    video_dir: str = "videos/ppo"
    video_episodes: int = 3


class ActorCritic(nn.Module):
    """MLP actor-critic with shared trunk and separate policy/value heads."""

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        # Initialize base nn.Module internals.
        super().__init__()
        # Shared representation network.
        # PPO commonly uses tanh activations for stable continuous optimization.
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        # Actor head: outputs action logits.
        self.policy_head = nn.Linear(hidden_size, n_actions)
        # Critic head: outputs scalar value estimate V(s).
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute shared latent representation.
        h = self.shared(x)
        # Return actor logits and critic values together.
        return self.policy_head(h), self.value_head(h)


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimates and bootstrapped returns."""

    # Advantages array aligned with rollout timesteps.
    advantages = np.zeros_like(rewards, dtype=np.float32)
    # Running GAE accumulator for backward recursion.
    gae = 0.0
    # Backward pass over rollout.
    for t in reversed(range(len(rewards))):
        # Mask used to stop bootstrap beyond terminal states.
        not_done = 1.0 - float(dones[t])
        # Value estimate of next state (bootstrap at end uses next_value arg).
        next_v = next_value if t == len(rewards) - 1 else values[t + 1]
        # TD residual delta_t.
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        # GAE recursion.
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
    # Return target for critic: R_t = A_t + V_t
    returns = advantages + values
    return advantages, returns


def run_ppo(config: PPOConfig | None = None) -> list[float]:
    """Train a PPO agent and return per-episode rewards."""

    # Resolve runtime config.
    cfg = config or PPOConfig()
    print("\n--- Starting PPO ---")

    # Build environment and infer dimensions.
    env = gym.make(cfg.env_name)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create actor-critic model and optimizer.
    net = ActorCritic(obs_size, n_actions, hidden_size=cfg.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    # Episode reward history used by benchmark metrics.
    rewards_history: list[float] = []
    # Initialize first episode state.
    state, _ = env.reset()
    # Running reward within current episode.
    ep_reward = 0.0

    # Train until target number of completed episodes is reached.
    while len(rewards_history) < cfg.episodes:
        # Rollout buffers for one PPO collection phase.
        states: list[np.ndarray] = []
        actions: list[int] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        values: list[float] = []

        # Collect fixed-length rollout under current policy.
        for _ in range(cfg.rollout_steps):
            # Convert state to batched tensor.
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # Inference step for policy + value.
                logits, value = net(state_t)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                # PPO collection uses stochastic action sampling.
                action_t = dist.sample()
                log_prob_t = dist.log_prob(action_t)

            # Step environment.
            action = int(action_t.item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append transition to rollout buffers.
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            log_probs.append(float(log_prob_t.item()))
            rewards.append(float(reward))
            dones.append(done)
            values.append(float(value.item()))

            # Update current episode accounting.
            ep_reward += float(reward)
            state = next_state

            if done:
                # Store completed episode reward for benchmark statistics.
                rewards_history.append(ep_reward)
                ep_reward = 0.0
                # Immediately begin next episode within same rollout window.
                state, _ = env.reset()
                if len(rewards_history) % 50 == 0:
                    avg_reward = float(np.mean(rewards_history[-50:]))
                    print(f"Episode {len(rewards_history)}, Average Reward (last 50): {avg_reward:.2f}")
                if len(rewards_history) >= cfg.episodes:
                    break

        # If rollout is empty (rare edge), skip update safely.
        if not states:
            continue

        # Bootstrap value for state immediately after last collected step.
        with torch.no_grad():
            next_state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            _, next_value_t = net(next_state_t)
            next_value = float(next_value_t.item())

        # Convert rollout lists to numpy arrays.
        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int64)
        old_log_probs_np = np.asarray(log_probs, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.bool_)
        values_np = np.asarray(values, dtype=np.float32)

        # Compute GAE advantages and value targets.
        adv_np, returns_np = _compute_gae(
            rewards_np,
            dones_np,
            values_np,
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        # Advantage normalization improves optimization stability.
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        # Convert arrays to tensors for minibatch updates.
        states_t = torch.tensor(states_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.int64)
        old_log_probs_t = torch.tensor(old_log_probs_np, dtype=torch.float32)
        adv_t = torch.tensor(adv_np, dtype=torch.float32)
        returns_t = torch.tensor(returns_np, dtype=torch.float32)

        # Build shuffled indices for SGD over rollout data.
        data_size = states_t.shape[0]
        idx = np.arange(data_size)

        # PPO performs multiple optimization passes over same rollout.
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, data_size, cfg.minibatch_size):
                # Slice minibatch indices.
                mb_idx = idx[start:start + cfg.minibatch_size]
                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # Forward pass on minibatch.
                logits, values_pred = net(mb_states)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                values_pred = values_pred.squeeze(-1)

                # Probability ratio between new and old policy.
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                # Unclipped surrogate objective.
                surr1 = ratio * mb_adv
                # Clipped surrogate objective.
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                # PPO policy loss uses pessimistic bound (min of two surrogates).
                policy_loss = -torch.min(surr1, surr2).mean()
                # Critic regression objective.
                value_loss = nn.functional.mse_loss(values_pred, mb_returns)

                # Combined PPO objective with entropy regularization.
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                # Gradient update step.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Optional post-training evaluation video.
    if cfg.record_video:
        net.eval()

        # Deterministic greedy policy for cleaner reproducible demos.
        def _policy(state: np.ndarray) -> int:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, _ = net(state_t)
                return int(torch.argmax(logits, dim=1).item())

        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="ppo",
            policy_fn=_policy,
        )

    # Cleanup environment resources.
    env.close()
    # Return completed-episode reward history.
    return rewards_history
