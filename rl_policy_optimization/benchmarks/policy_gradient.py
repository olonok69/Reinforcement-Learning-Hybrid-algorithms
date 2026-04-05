from __future__ import annotations
"""Policy Gradient (REINFORCE) benchmark on CartPole-v1.

This module provides a compact, reproducible baseline for direct policy
optimization using Monte Carlo returns.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.common import record_policy_video


@dataclass
class PolicyGradientConfig:
    """Configuration for REINFORCE training."""

    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    learning_rate: float = 1e-3
    episodes: int = 700
    hidden_size: int = 128
    normalize_returns: bool = True
    record_video: bool = False
    video_dir: str = "videos/policy_gradient"
    video_episodes: int = 3


class PolicyNetwork(nn.Module):
    """Small MLP policy that outputs action logits."""

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        # Initialize nn.Module internals (parameter registration, hooks, etc.).
        super().__init__()
        # Build a small MLP that maps state -> action logits.
        # Layout:
        #   input_size -> hidden_size -> 128 -> n_actions
        # ReLU adds non-linearity so the network can learn non-linear decision rules.
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the MLP. Output is raw logits (not probabilities yet).
        return self.net(x)


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted returns for one episode."""

    # Will store G_t values in reverse first, then flip at the end.
    out: list[float] = []
    # Running accumulator for return-from-current-step.
    running = 0.0
    # Iterate backward so each step can reuse return from next step:
    # G_t = r_t + gamma * G_{t+1}
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    # Reverse to align returns with original time order (t=0 ... T-1).
    out.reverse()
    return out


def run_policy_gradient(config: PolicyGradientConfig | None = None) -> list[float]:
    """Train a REINFORCE agent and return per-episode rewards."""

    # Use provided config or default hyperparameters.
    cfg = config or PolicyGradientConfig()
    print("\n--- Starting Policy Gradient (REINFORCE) ---")

    # Create the Gymnasium environment.
    env = gym.make(cfg.env_name)
    # CartPole observation space is a 1D vector; shape[0] is feature count.
    obs_size = env.observation_space.shape[0]
    # Number of discrete actions (left/right for CartPole).
    n_actions = env.action_space.n

    # Instantiate policy network and optimizer.
    net = PolicyNetwork(obs_size, n_actions, hidden_size=cfg.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    # Keep total reward per episode for downstream benchmark metrics.
    episode_rewards: list[float] = []

    # Main REINFORCE training loop.
    for episode in range(cfg.episodes):
        # Reset environment to start new episode.
        state, _ = env.reset()
        # Episode termination flag.
        done = False

        # Per-episode storage:
        # - log_probs for selected actions (policy gradient term)
        # - rewards to compute Monte Carlo returns
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []

        # Rollout one full episode with current stochastic policy.
        while not done:
            # Convert numpy state -> float tensor and add batch dimension [1, obs_size].
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # Network outputs action logits. (vector of length n_actions in this case 2 for CartPole)
            logits = net(state_t)
            # Convert logits to action probabilities.
            probs = torch.softmax(logits, dim=1)
            # Build categorical distribution to sample discrete action.
            dist = torch.distributions.Categorical(probs)
            # Sample an action (stochastic policy behavior).(here  yo have 2 actions in CartPole: left or right and you sample one of them according to the probabilities output by the network)
            # Note: sampling is essential for exploration and correct policy gradient estimation.)
            action = dist.sample() # here is where we do our exploration by sampling from the action distribution output by the policy network

            # Step environment with sampled action.
            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            # Gymnasium episode ends if either terminated or truncated.
            done = terminated or truncated

            # Store log π(a_t|s_t) for policy gradient.
            log_probs.append(dist.log_prob(action))
            # Store reward for Monte Carlo return computation later.
            rewards.append(float(reward))
            # Move to next state.
            state = next_state

        # Compute reward-to-go returns for every timestep in the episode.
        returns = _discounted_returns(rewards, cfg.gamma)
        # Convert returns to tensor for vectorized loss math.
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Optional variance/stability trick:
        # normalize returns to mean 0 / std 1 (only if at least 2 steps).
        if cfg.normalize_returns and len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # REINFORCE objective (minimization form):
        # loss = -Σ logπ(a_t|s_t) * G_t
        # minus sign converts gradient-descent optimizer into ascent on expected return.
        policy_loss = -torch.stack(log_probs) * returns_t
        loss = policy_loss.sum()

        # Standard PyTorch optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Episode total reward for monitoring/benchmarking.
        ep_reward = float(np.sum(rewards))
        episode_rewards.append(ep_reward)

        # Print progress every 50 episodes.
        if (episode + 1) % 50 == 0:
            avg_reward = float(np.mean(episode_rewards[-50:]))
            print(f"Episode {episode + 1}, Average Reward (last 50): {avg_reward:.2f}")

    # Optional post-training evaluation video recording.
    if cfg.record_video:
        # Evaluation mode disables training-time behavior (e.g., dropout/bn if present).
        net.eval()

        # Deterministic greedy policy for clean evaluation videos.
        def _policy(state: np.ndarray) -> int:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = net(state_t)
                return int(torch.argmax(logits, dim=1).item())

        # Shared helper wraps env with RecordVideo and saves episodes to disk.
        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="policy_gradient",
            policy_fn=_policy,
        )

    # Release environment resources explicitly.
    env.close()
    # Return reward history so benchmark runner can compute metrics.
    return episode_rewards
