"""td3 — Twin Delayed Deep Deterministic Policy Gradient (TD3) for continuous control.

TD3 (Fujimoto et al. 2018) is a direct improvement over DDPG that addresses its three
main failure modes with three targeted fixes:

1. **Twin critics** — two independent Q-networks (``Q1``, ``Q2``) whose *minimum* is used
   for the Bellman target.  This pessimistic target counters Q-value overestimation, which
   occurs because the actor always pushes toward the current maximum of the critic.

2. **Delayed actor updates** — the actor (and all target networks) are updated only every
   ``policy_delay`` critic updates.  Updating the actor less frequently gives the critics
   time to converge before their gradients are used to drive policy improvement.

3. **Target policy smoothing** — clipped Gaussian noise is added to the target actor's
   action before evaluating the target critics.  This regularises the Q surface by
   averaging over a neighbourhood of actions, preventing the actor from exploiting sharp
   local peaks in the critic.

The :class:`Actor` architecture is identical to DDPG.  All three fixes live in
:class:`TwinCritic` and the update block of :func:`run_td3`.

Reference: https://arxiv.org/abs/1802.09477
"""

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
class TD3Config:
    """Hyperparameters for a TD3 training run.

    Extends :class:`~benchmarks.ddpg.DDPGConfig` with three fields that implement
    TD3's stability improvements.  Shared fields (``gamma``, ``tau``, etc.) have
    the same semantics as in DDPG.

    Attributes
    ----------
    env_name:
        Gymnasium environment ID.
    episodes:
        Total number of training episodes.
    max_steps:
        Maximum environment steps per episode.
    gamma:
        Discount factor.
    tau:
        Polyak soft-update coefficient for target networks.
    actor_lr:
        Adam learning rate for the actor.  Lower than DDPG default (``3e-4`` vs ``1e-3``)
        to match the delayed, less frequent updates.
    critic_lr:
        Adam learning rate for the critics.
    hidden_size:
        Hidden layer width for all networks.
    batch_size:
        Mini-batch size per gradient update.
    buffer_size:
        Replay buffer capacity.
    warmup_steps:
        Random-action steps before the first gradient update.
    exploration_noise:
        Std of Gaussian noise added to the online actor during *data collection*.
        Distinct from ``policy_noise``, which applies only to the *target* actor.
    policy_noise:
        Std of Gaussian noise added to the *target* actor's action (Fix 3 — target
        policy smoothing).  Expressed as a fraction of ``max_action``.
    noise_clip:
        Maximum absolute value of the target policy noise, expressed as a fraction of
        ``max_action``.  Acts as a second safety bound on the smoothing noise.
    policy_delay:
        Number of critic updates between consecutive actor updates (Fix 2 — delayed
        actor updates).  ``2`` means the actor updates half as often as the critics.
    train_updates_per_step:
        Gradient update cycles per environment step.
    record_video:
        Record evaluation episodes after training if ``True``.
    video_dir:
        Output directory for video files.
    video_episodes:
        Number of episodes to record.
    """

    env_name: str = "Pendulum-v1"
    episodes: int = 160
    max_steps: int = 200
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_size: int = 256
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup_steps: int = 5_000
    exploration_noise: float = 0.1
    policy_noise: float = 0.2     # Fix 3: target smoothing noise std
    noise_clip: float = 0.5       # Fix 3: target smoothing noise magnitude cap
    policy_delay: int = 2         # Fix 2: critic updates per actor update
    train_updates_per_step: int = 1
    record_video: bool = False
    video_dir: str = "videos/td3"
    video_episodes: int = 3


class Actor(nn.Module):
    """Deterministic policy network: state → action.

    Identical architecture to the DDPG :class:`~benchmarks.ddpg.Actor`.
    The ``tanh`` output squashes activations to ``(-1, 1)`` and ``max_action``
    rescales to the environment's action bounds.

    Parameters
    ----------
    state_dim:
        Dimensionality of the input state vector.
    action_dim:
        Dimensionality of the output action vector.
    hidden_size:
        Number of neurons in each hidden layer.
    max_action:
        Action space upper bound used for output rescaling.
    """

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
        """Map a batch of states to deterministic actions.

        Parameters
        ----------
        state:
            Tensor of shape ``(batch, state_dim)``.

        Returns
        -------
        torch.Tensor
            Action tensor of shape ``(batch, action_dim)`` in
            ``[-max_action, max_action]``.
        """
        return torch.tanh(self.net(state)) * self.max_action


class TwinCritic(nn.Module):
    """Two independent Q-function networks sharing the same interface.

    **Fix 1 — Twin critics (clipped double-Q):** Having two critics with different random
    initialisations produces independent, slightly diverse Q-estimates.  Using the minimum
    of the two as the Bellman target is a pessimistic estimator that counteracts the
    systematic upward bias introduced when the actor continually maximises a noisy critic.

    Both ``q1`` and ``q2`` are standard MLP critics that take ``(state, action)`` as
    their joint input.

    Parameters
    ----------
    state_dim:
        Dimensionality of the state vector.
    action_dim:
        Dimensionality of the action vector.
    hidden_size:
        Number of neurons in each hidden layer for both heads.
    """

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
        """Return Q-values from both critic heads for the same input.

        Parameters
        ----------
        state:
            Tensor of shape ``(batch, state_dim)``.
        action:
            Tensor of shape ``(batch, action_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(q1_values, q2_values)``, each of shape ``(batch, 1)``.
        """
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only the ``Q1`` estimate (used exclusively for the actor update).

        The actor update uses ``Q1`` alone — not ``min(Q1, Q2)`` — to avoid the actor
        optimising against the pessimistic lower bound, which would under-estimate the
        policy's true value and slow learning.

        Parameters
        ----------
        state:
            Tensor of shape ``(batch, state_dim)``.
        action:
            Tensor of shape ``(batch, action_dim)``.

        Returns
        -------
        torch.Tensor
            Q1 value tensor of shape ``(batch, 1)``.
        """
        return self.q1(torch.cat([state, action], dim=1))


def run_td3(config: TD3Config | None = None) -> list[float]:
    """Train a TD3 agent and return the per-episode reward history.

    The training loop is identical to DDPG in structure, with three modifications
    inside the gradient update block:

    * **Twin-critic target** (Fix 1): Bellman target uses ``min(Q1_target, Q2_target)``.
    * **Target smoothing** (Fix 3): clipped Gaussian noise on the target actor's action
      before evaluating the target critics.
    * **Delayed actor update** (Fix 2): actor and target networks are updated only every
      ``policy_delay`` critic gradient steps.

    Parameters
    ----------
    config:
        Hyperparameter configuration.  Defaults to :class:`TD3Config` if ``None``.

    Returns
    -------
    list[float]
        Total undiscounted reward per episode in chronological order.
    """
    cfg = config or TD3Config()
    print("\n--- Starting TD3 ---")

    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Networks ---
    actor = Actor(state_dim, action_dim, cfg.hidden_size, max_action).to(device)
    actor_target = Actor(state_dim, action_dim, cfg.hidden_size, max_action).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # TwinCritic holds both Q1 and Q2; critic_target is only soft-updated.
    critic = TwinCritic(state_dim, action_dim, cfg.hidden_size).to(device)
    critic_target = TwinCritic(state_dim, action_dim, cfg.hidden_size).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    replay = ReplayBuffer(state_dim, action_dim, max_size=cfg.buffer_size, device=device)

    rewards_history: list[float] = []
    global_step = 0  # env steps across all episodes
    update_step = 0  # gradient update steps (used to gate actor updates)

    for episode in range(cfg.episodes):
        state, _ = env.reset()
        ep_reward = 0.0

        for _ in range(cfg.max_steps):
            global_step += 1
            state_arr = np.asarray(state, dtype=np.float32)

            # --- Action selection ---
            if global_step < cfg.warmup_steps:
                # Warmup: random actions to populate the buffer.
                action = env.action_space.sample()
            else:
                # Training: deterministic actor + exploration noise for data collection.
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

            # --- Gradient update ---
            if replay.size >= max(cfg.warmup_steps, cfg.batch_size):
                for _ in range(cfg.train_updates_per_step):
                    update_step += 1
                    s, a, ns, r, nd = replay.sample(cfg.batch_size)

                    # --- Fix 3: Target policy smoothing ---
                    # Add clipped Gaussian noise to the target actor's action.
                    # The double clamp: first on the noise magnitude, then on the full
                    # action, prevents extreme target values.
                    with torch.no_grad():
                        noise = (torch.randn_like(a) * cfg.policy_noise * max_action).clamp(
                            -cfg.noise_clip * max_action,   # inner clip: cap noise size
                            cfg.noise_clip * max_action,
                        )
                        next_action = (actor_target(ns) + noise).clamp(-max_action, max_action)

                        # --- Fix 1: Clipped double-Q target ---
                        # Use the pessimistic minimum of both target critics.
                        target_q1, target_q2 = critic_target(ns, next_action)
                        target_q = r + nd * cfg.gamma * torch.min(target_q1, target_q2)

                    # Critic update: both Q1 and Q2 train toward the same target.
                    # Independent initialisations cause them to diverge slightly over time,
                    # which is what makes their minimum a useful pessimistic estimator.
                    current_q1, current_q2 = critic(s, a)
                    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    # --- Fix 2: Delayed actor + target-network update ---
                    # Actor and targets only update every policy_delay critic steps.
                    if update_step % cfg.policy_delay == 0:
                        # Actor uses Q1 only (not the pessimistic minimum).
                        actor_loss = -critic.q1_only(s, actor(s)).mean()

                        actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_opt.step()

                        # Soft update targets only when the actor updates, keeping all
                        # three networks (actor, Q1, Q2 targets) advancing in sync.
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

    # --- Optional post-training video recording ---
    if cfg.record_video:
        actor.eval()

        def _policy(state_np: np.ndarray) -> np.ndarray:
            """Deterministic policy wrapper for video recording (no exploration noise)."""
            with torch.no_grad():
                s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                a = actor(s).squeeze(0).cpu().numpy()
            return np.clip(a, env.action_space.low, env.action_space.high)

        record_policy_video_continuous(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="td3",
            policy_fn=_policy,
        )

    env.close()
    return rewards_history
