"""ddpg — Deep Deterministic Policy Gradient (DDPG) for continuous control.

DDPG (Lillicrap et al. 2015) extends DQN to continuous action spaces by learning
a deterministic policy (actor) alongside a Q-function (critic).  Key components:

* **Deterministic actor** ``mu(s)`` — outputs a single action vector directly,
  avoiding the intractable ``argmax`` over continuous actions.
* **Critic** ``Q(s, a)`` — evaluates (state, action) pairs; its gradient w.r.t.
  ``a`` is used to update the actor via the deterministic policy gradient theorem.
* **Target networks** — slowly-tracking copies of both actor and critic that
  provide stable Bellman bootstrap targets (Polyak / soft update, ``tau=0.005``).
* **Replay buffer** — stores off-policy transitions and breaks temporal correlation
  between gradient updates (see :mod:`benchmarks.replay_buffer`).
* **Exploration noise** — Gaussian noise added to the deterministic action during
  training; DDPG has no intrinsic exploration mechanism.

Reference: https://arxiv.org/abs/1509.02971
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
class DDPGConfig:
    """Hyperparameters for a DDPG training run.

    All fields have sensible defaults for ``Pendulum-v1`` and can be overridden
    either programmatically or via the ``ddpg_benchmark.py`` CLI flags.

    Attributes
    ----------
    env_name:
        Gymnasium environment ID.
    episodes:
        Total number of training episodes.
    max_steps:
        Maximum environment steps per episode.  Pendulum-v1 caps naturally at 200.
    gamma:
        Discount factor for future rewards.
    tau:
        Polyak soft-update coefficient for target networks.  ``tau=0.005`` causes
        target parameters to trail the online parameters at 0.5 % per step, providing
        stable Bellman targets without freezing them.
    actor_lr:
        Adam learning rate for the actor network.
    critic_lr:
        Adam learning rate for the critic network.
    hidden_size:
        Number of neurons in each hidden layer (both actor and critic use the same).
    batch_size:
        Mini-batch size sampled from the replay buffer per gradient update.
    buffer_size:
        Maximum capacity of the replay buffer (older transitions are overwritten).
    warmup_steps:
        Number of random-action steps taken before the first gradient update.
        Ensures the buffer holds diverse data before learning begins.
    exploration_noise:
        Standard deviation of the Gaussian exploration noise added to the
        deterministic action, expressed as a fraction of ``max_action``.
        ``0.1`` → 10 % of the action range as 1-sigma noise.
    train_updates_per_step:
        Number of gradient update cycles per environment step.  ``1`` is standard.
    record_video:
        If ``True``, record evaluation episodes to ``video_dir`` after training.
    video_dir:
        Directory for recorded video files.  Created automatically.
    video_episodes:
        Number of deterministic evaluation episodes to record.
    """

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
    """Deterministic policy network: state → action.

    Architecture: Linear(state_dim → hidden) → ReLU → Linear → ReLU → Linear(→ action_dim).
    The final ``tanh`` squashes pre-activations to ``(-1, 1)`` and ``max_action`` rescales
    the output to the environment's actual action bounds.  This guarantees the actor output
    is always within the valid range without explicit clipping in the forward pass.

    Parameters
    ----------
    state_dim:
        Dimensionality of the input observation vector.
    action_dim:
        Dimensionality of the output action vector.
    hidden_size:
        Number of neurons in each of the two hidden layers.
    max_action:
        Action space upper bound (``env.action_space.high[0]``).  Multiplied by
        ``tanh`` output to rescale to ``[-max_action, max_action]``.
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


class Critic(nn.Module):
    """Q-function network: (state, action) → scalar Q-value.

    Takes the concatenation of state and action as a single input vector.
    This joint representation allows the gradient ``dQ/da`` to flow through the
    critic into the actor during the actor update step (deterministic policy gradient).

    Architecture: Linear(state_dim + action_dim → hidden) → ReLU → Linear → ReLU → Linear(→ 1).

    Parameters
    ----------
    state_dim:
        Dimensionality of the state observation.
    action_dim:
        Dimensionality of the action vector.
    hidden_size:
        Number of neurons in each hidden layer.
    """

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
        """Compute the Q-value for a batch of (state, action) pairs.

        Parameters
        ----------
        state:
            Tensor of shape ``(batch, state_dim)``.
        action:
            Tensor of shape ``(batch, action_dim)``.

        Returns
        -------
        torch.Tensor
            Q-value tensor of shape ``(batch, 1)``.
        """
        return self.net(torch.cat([state, action], dim=1))


def run_ddpg(config: DDPGConfig | None = None) -> list[float]:
    """Train a DDPG agent and return the per-episode reward history.

    Implements the standard DDPG training loop:

    1. Warmup: fill the replay buffer with ``warmup_steps`` random-action transitions.
    2. Per step: select action = ``actor(s) + Gaussian noise``, store transition, then
       (once buffer is large enough) perform one gradient update cycle:

       a. Sample mini-batch ``(s, a, s', r, not_done)`` from the replay buffer.
       b. Compute Bellman target using *frozen* target networks:
          ``y = r + not_done * gamma * Q_target(s', actor_target(s'))``.
       c. Update critic: minimise ``MSE(Q(s,a), y)``.
       d. Update actor: maximise ``Q(s, actor(s))`` (minimise its negative mean).
       e. Soft-update both target networks (Polyak averaging with ``tau``).

    3. Optional: record evaluation video after the training loop.

    Parameters
    ----------
    config:
        Hyperparameter configuration.  Defaults to :class:`DDPGConfig` if ``None``.

    Returns
    -------
    list[float]
        Total undiscounted reward for each episode, in chronological order.
        Can be passed directly to :func:`~benchmarks.common.run_timed` or used for
        custom analysis.
    """
    cfg = config or DDPGConfig()
    print("\n--- Starting DDPG ---")

    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Networks ---
    actor = Actor(state_dim, action_dim, cfg.hidden_size, max_action).to(device)
    # Target actor starts as an exact copy; never trained directly — only soft-updated.
    actor_target = Actor(state_dim, action_dim, cfg.hidden_size, max_action).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, action_dim, cfg.hidden_size).to(device)
    # Target critic starts as an exact copy; provides stable Bellman bootstrap values.
    critic_target = Critic(state_dim, action_dim, cfg.hidden_size).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    replay = ReplayBuffer(state_dim, action_dim, max_size=cfg.buffer_size, device=device)

    rewards_history: list[float] = []
    global_step = 0  # counts total env steps across all episodes

    for episode in range(cfg.episodes):
        state, _ = env.reset()
        ep_reward = 0.0

        for _ in range(cfg.max_steps):
            global_step += 1
            # Convert state to a NumPy array of type float32 for the networks and replay buffer.
            state_arr = np.asarray(state, dtype=np.float32)

            # --- Action selection ---
            if global_step < cfg.warmup_steps:
                # Warmup phase: pure random actions seed the buffer with diverse data.
                action = env.action_space.sample()
            else:
                # Training phase: deterministic actor + additive Gaussian exploration noise.
                with torch.no_grad():
                    state_t = torch.tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)
                    action = actor(state_t).squeeze(0).cpu().numpy()
                noise = np.random.normal(0.0, cfg.exploration_noise * max_action, size=action_dim)
                # Clip to action bounds after noise addition.
                action = np.clip(action + noise, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition; not_done = 1 - done is handled inside ReplayBuffer.add.
            replay.add(
                state_arr,
                np.asarray(action, dtype=np.float32),
                np.asarray(next_state, dtype=np.float32),
                float(reward),
                bool(done),
            )

            state = next_state
            ep_reward += float(reward)

            # --- Gradient update (skipped until buffer has enough transitions) ---
            if replay.size >= max(cfg.warmup_steps, cfg.batch_size):
                for _ in range(cfg.train_updates_per_step):
                    # s, a, ns, r, nd shapes: (batch, dim) tensors already on device.
                    s, a, ns, r, nd = replay.sample(cfg.batch_size)

                    # Step 1 — Bellman target (no gradients through target networks).
                    with torch.no_grad():
                        next_action = actor_target(ns)
                        target_q = critic_target(ns, next_action)
                        # nd = (1 - done); zeros out the bootstrap when episode ends.
                        target_q = r + nd * cfg.gamma * target_q

                    # Step 2 — Critic update: minimise MSE between Q(s,a) and the target.
                    current_q = critic(s, a)
                    critic_loss = F.mse_loss(current_q, target_q)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    # Step 3 — Actor update: maximise Q(s, actor(s)) via the critic gradient.
                    # Gradient chain: actor_loss → Q(s, actor(s)) → actor(s) → actor weights.
                    actor_loss = -critic(s, actor(s)).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # Step 4 — Polyak soft update of both target networks.
                    # tp ← (1 - tau) * tp + tau * p  (in-place, no new tensors allocated)
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
            name_prefix="ddpg",
            policy_fn=_policy,
        )

    env.close()
    return rewards_history
