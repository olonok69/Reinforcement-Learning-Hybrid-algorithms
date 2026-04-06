from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.common import record_policy_video_continuous
from benchmarks.replay_buffer import ReplayBuffer


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class SACConfig:
    env_name: str = "Pendulum-v1"
    episodes: int = 160
    max_steps: int = 200
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha: float = 0.2
    hidden_size: int = 256
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup_steps: int = 5_000
    train_updates_per_step: int = 1
    record_video: bool = False
    video_dir: str = "videos/sac"
    video_episodes: int = 3


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, max_action: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()

        if deterministic:
            z = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()

        squashed = torch.tanh(z)
        action = squashed * self.max_action

        if deterministic:
            log_prob = torch.zeros((state.shape[0], 1), device=state.device, dtype=state.dtype)
        else:
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(z)
            correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
            log_prob = (log_prob - correction).sum(dim=1, keepdim=True)

        return action, log_prob


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


def run_sac(config: SACConfig | None = None) -> list[float]:
    cfg = config or SACConfig()
    print("\n--- Starting SAC ---")

    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = GaussianActor(state_dim, action_dim, cfg.hidden_size, max_action).to(device)
    critic = TwinCritic(state_dim, action_dim, cfg.hidden_size).to(device)
    critic_target = TwinCritic(state_dim, action_dim, cfg.hidden_size).to(device)
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
                    s = torch.tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)
                    action_t, _ = actor.sample(s, deterministic=False)
                    action = action_t.squeeze(0).cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)

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
                        next_a, next_logp = actor.sample(ns, deterministic=False)
                        target_q1, target_q2 = critic_target(ns, next_a)
                        target_q = torch.min(target_q1, target_q2) - cfg.alpha * next_logp
                        target = r + nd * cfg.gamma * target_q

                    current_q1, current_q2 = critic(s, a)
                    critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    pi_action, logp_pi = actor.sample(s, deterministic=False)
                    q1_pi, q2_pi = critic(s, pi_action)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (cfg.alpha * logp_pi - min_q_pi).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    with torch.no_grad():
                        for p, tp in zip(critic.parameters(), critic_target.parameters()):
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
                a, _ = actor.sample(s, deterministic=True)
                action = a.squeeze(0).cpu().numpy()
            return np.clip(action, env.action_space.low, env.action_space.high)

        record_policy_video_continuous(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="sac",
            policy_fn=_policy,
        )

    env.close()
    return rewards_history
