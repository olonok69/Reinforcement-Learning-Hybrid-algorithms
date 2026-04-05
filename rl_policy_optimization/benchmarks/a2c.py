from __future__ import annotations
"""Advantage Actor-Critic (A2C) benchmark on CartPole-v1."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.common import record_policy_video


@dataclass
class A2CConfig:
    """Configuration for A2C training."""

    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    learning_rate: float = 1e-3
    episodes: int = 500
    hidden_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    record_video: bool = False
    video_dir: str = "videos/a2c"
    video_episodes: int = 3


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        # Initialize nn.Module internals (parameter registration, etc.).
        super().__init__()
        # Shared feature extractor used by both actor and critic.
        # A shared trunk reduces parameters and lets both heads learn
        # a common state representation.
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        # Actor head: maps shared features -> action logits.
        self.policy_head = nn.Linear(hidden_size, n_actions)
        # Critic head: maps shared features -> scalar state value V(s).
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Build latent features from raw state input.
        features = self.shared(x)
        # Actor output (unnormalized action scores).
        logits = self.policy_head(features)
        # Critic output (value estimate).
        value = self.value_head(features)
        # Return both so the training loop can compute policy + value losses.
        return logits, value


def run_a2c(config: A2CConfig | None = None) -> list[float]:
    """Train an A2C agent and return per-episode rewards."""

    # Use caller config if provided; otherwise fall back to defaults.
    cfg = config or A2CConfig()
    print("\n--- Starting A2C ---")

    # Create the Gymnasium environment instance.
    env = gym.make(cfg.env_name)
    # Observation vector size (CartPole has 4 features).
    obs_size = env.observation_space.shape[0]
    # Number of discrete actions (CartPole has 2 actions).
    n_actions = env.action_space.n

    # Build actor-critic network and optimizer.
    net = ActorCritic(obs_size, n_actions, hidden_size=cfg.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    # Per-episode reward history for benchmark metrics.
    episode_rewards: list[float] = []

    # Main training loop over episodes.
    for episode in range(cfg.episodes):
        # Reset environment at the start of each episode.
        state, _ = env.reset()
        # Episode terminates when either terminated or truncated is True.
        done = False

        # Trajectory buffers for one episode.
        # log_probs: log π(a_t|s_t) terms for actor loss
        # values: critic predictions V(s_t)
        # rewards: environment rewards
        # entropies: policy entropy for exploration regularization
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        entropies: list[torch.Tensor] = []

        # Roll out one full on-policy episode.
        while not done:
            # Convert state ndarray -> torch tensor with batch dimension [1, obs].
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # Forward pass gives policy logits and state value estimate.
            logits, value = net(state_t)

            # Convert logits to action probabilities.
            probs = torch.softmax(logits, dim=1)
            # Categorical distribution for discrete action sampling.
            dist = torch.distributions.Categorical(probs)
            # Sample action stochastically from current policy.
            action = dist.sample()

            # Step environment with sampled action.
            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            # Gym episode end condition.
            done = terminated or truncated

            # Store trajectory elements used by loss computation later.
            log_probs.append(dist.log_prob(action))
            # Value head returns [1, 1]; squeeze to scalar tensor.
            values.append(value.squeeze(-1).squeeze(0))
            rewards.append(float(reward))
            # Entropy encourages non-collapsed action distributions.
            entropies.append(dist.entropy())
            # Move to the next state.
            state = next_state

        # Compute Monte Carlo discounted returns G_t for the episode.
        returns: list[float] = []
        running_return = 0.0
        # Backward recursion: G_t = r_t + gamma * G_{t+1}
        for reward in reversed(rewards):
            running_return = reward + cfg.gamma * running_return
            returns.append(running_return)
        # Returns were built backward, restore time order.
        returns.reverse()

        # Convert trajectory lists to tensors for vectorized math.
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        # Average entropy over episode steps.
        entropy_t = torch.stack(entropies).mean()

        # Advantage estimate: A_t = G_t - V(s_t).
        # Detach values so actor gradient does not flow through critic branch.
        advantages_t = returns_t - values_t.detach()
        # Actor objective (negative because optimizer minimizes loss).
        policy_loss = -(log_probs_t * advantages_t).mean()
        # Critic objective: fit value predictions to Monte Carlo returns.
        value_loss = nn.functional.mse_loss(values_t, returns_t)

        # Combined A2C loss with entropy regularization:
        # - Policy term improves action selection.
        # - Value term improves baseline quality.
        # - Entropy term avoids premature deterministic collapse.
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_t

        # Standard backward/update step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Episode score tracking.
        ep_reward = float(np.sum(rewards))
        episode_rewards.append(ep_reward)

        # Periodic progress logging.
        if (episode + 1) % 50 == 0:
            avg_reward = float(np.mean(episode_rewards[-50:]))
            print(f"Episode {episode + 1}, Average Reward (last 50): {avg_reward:.2f}")

    # Optional post-training video recording.
    if cfg.record_video:
        # Evaluation mode for deterministic inference behavior.
        net.eval()

        # Greedy policy used for cleaner/consistent evaluation videos.
        def _policy(state: np.ndarray) -> int:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, _ = net(state_t)
                return int(torch.argmax(logits, dim=1).item())

        # Shared helper handles RecordVideo wrapper and saving artifacts.
        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="a2c",
            policy_fn=_policy,
        )

    # Release environment resources.
    env.close()
    # Return reward history for benchmark summary computation.
    return episode_rewards
