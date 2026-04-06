from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.common import record_policy_video_continuous
from benchmarks.replay_buffer import ReplayBuffer


@dataclass
class DDPGConfig:
    env_name: str = "Pendulum-v1"
    episodes: int = 160
    max_steps: int = 200
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden_size: int = 256
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup_steps: int = 5_000
    exploration_noise: float = 0.1
    train_updates_per_step: int = 1
    record_video: bool = False
    video_dir: str = "videos/ddpg"
    video_episodes: int = 3


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, max_action: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state)) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))


def run_ddpg(config: DDPGConfig | None = None) -> list[float]:
    cfg = config or DDPGConfig()
    print("\n--- Starting DDPG ---")

    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor(state_dim, action_dim, cfg.hidden_size, max_action).to(device)
    actor_target = Actor(state_dim, action_dim, cfg.hidden_size, max_action).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, action_dim, cfg.hidden_size).to(device)
    critic_target = Critic(state_dim, action_dim, cfg.hidden_size).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    replay = ReplayBuffer(state_dim, action_dim, max_size=cfg.buffer_size, device=device)

    rewards_history: list[float] = []
    global_step = 0

    for episode in range(cfg.episodes):
        state, _ = env.reset()
        ep_reward = 0.0

        for _ in range(cfg.max_steps):
            global_step += 1

            state_arr = np.asarray(state, dtype=np.float32)
            if global_step < cfg.warmup_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)
                    action = actor(state_t).squeeze(0).cpu().numpy()
                noise = np.random.normal(0.0, cfg.exploration_noise * max_action, size=action_dim)
                action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.add(
                state_arr,
                np.asarray(action, dtype=np.float32),
                np.asarray(next_state, dtype=np.float32),
                float(reward),
                bool(done),
            )

            state = next_state
            ep_reward += float(reward)

            if replay.size >= max(cfg.warmup_steps, cfg.batch_size):
                for _ in range(cfg.train_updates_per_step):
                    s, a, ns, r, nd = replay.sample(cfg.batch_size)

                    with torch.no_grad():
                        next_action = actor_target(ns)
                        target_q = critic_target(ns, next_action)
                        target_q = r + nd * cfg.gamma * target_q

                    current_q = critic(s, a)
                    critic_loss = F.mse_loss(current_q, target_q)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    actor_loss = -critic(s, actor(s)).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    with torch.no_grad():
                        for p, tp in zip(critic.parameters(), critic_target.parameters()):
                            tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)
                        for p, tp in zip(actor.parameters(), actor_target.parameters()):
                            tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)

            if done:
                break

        rewards_history.append(ep_reward)
        if (episode + 1) % 10 == 0:
            avg10 = float(np.mean(rewards_history[-10:]))
            print(f"Episode {episode + 1}/{cfg.episodes}, Avg Reward (last 10): {avg10:.2f}")

    if cfg.record_video:
        actor.eval()

        def _policy(state_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                a = actor(s).squeeze(0).cpu().numpy()
            return np.clip(a, env.action_space.low, env.action_space.high)

        record_policy_video_continuous(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="ddpg",
            policy_fn=_policy,
        )

    env.close()
    return rewards_history
