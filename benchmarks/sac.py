"""sac — Soft Actor-Critic (SAC) for continuous control.

SAC (Haarnoja et al. 2018) is an off-policy, entropy-regularised actor-critic algorithm.
It augments the standard discounted return objective with a weighted entropy bonus:

    J = E[Σ γᵗ (rₜ + α H(π(·|sₜ)))]

where ``H(π(·|sₜ)) = -E[log π(aₜ|sₜ)]`` is the policy entropy and ``α`` is the
temperature parameter controlling the entropy–reward trade-off.

**Key design choices that differ from DDPG/TD3:**

* **Stochastic policy** — :class:`GaussianActor` outputs a Gaussian distribution over
  actions (mean + log-std), sampled via the reparameterisation trick.  The stochastic
  policy naturally regularises the Q surface, removing the need for target policy
  smoothing (TD3 Fix 3) or a dedicated actor target network.
* **No actor target network** — because the stochastic actor is its own regulariser,
  SAC only maintains critic target networks.
* **Entropy in the critic target** — the Bellman backup includes ``-α * log π(a'|s')``
  in addition to the reward, so the critic learns to value states where the policy can
  act both well and diversely.
* **Dual critics (clipped double-Q)** — same idea as TD3 Fix 1: the minimum of two
  independent Q-estimates is used for Bellman targets.
* **Tanh squashing with log-prob correction** — the Gaussian sample is squashed through
  ``tanh`` to bound actions; the log-prob must be corrected for the Jacobian of the
  ``tanh`` transform (see :meth:`GaussianActor.sample`).

Reference: https://arxiv.org/abs/1801.01290
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


# Bounds on the log-standard-deviation output of the actor network.
# Clamping prevents exp(log_std) from collapsing to 0 (LOG_STD_MIN) or
# producing NaN/Inf values (LOG_STD_MAX).
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class SACConfig:
    """Hyperparameters for a SAC training run.

    Attributes
    ----------
    env_name:
        Gymnasium environment ID.
    episodes:
        Total number of training episodes.
    max_steps:
        Maximum environment steps per episode.
    gamma:
        Discount factor for the Bellman backup.
    tau:
        Polyak soft-update coefficient for the critic target networks.
    actor_lr:
        Adam learning rate for the actor.
    critic_lr:
        Adam learning rate for both critics.
    alpha:
        Entropy temperature coefficient.  Higher values encourage more exploration
        by placing greater weight on the entropy bonus.  ``0.2`` is a good starting
        point for most continuous control tasks.
    hidden_size:
        Hidden layer width shared across all networks.
    batch_size:
        Mini-batch size per gradient update.
    buffer_size:
        Replay buffer capacity.
    warmup_steps:
        Environment steps of random exploration before the first gradient update.
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
    alpha: float = 0.2              # entropy temperature
    hidden_size: int = 256
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup_steps: int = 5_000
    train_updates_per_step: int = 1
    record_video: bool = False
    video_dir: str = "videos/sac"
    video_episodes: int = 3


class GaussianActor(nn.Module):
    """Stochastic policy network: state → Gaussian distribution over actions.

    The network has a shared two-layer backbone that feeds two parallel linear heads:
    one for the predicted ``mean`` and one for the predicted ``log_std``.

    Actions are sampled via the **reparameterisation trick** (``rsample()``) and then
    squashed through ``tanh`` to respect the environment action bounds.  Because the
    ``tanh`` transform changes the probability density, the log-probability must be
    corrected accordingly (see :meth:`sample`).

    Parameters
    ----------
    state_dim:
        Dimensionality of the input state vector.
    action_dim:
        Dimensionality of the output action vector.
    hidden_size:
        Number of neurons in each hidden layer.
    max_action:
        Action space upper bound used to rescale the ``tanh`` output.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, max_action: float):
        super().__init__()
        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Separate heads for the distribution parameters
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the Gaussian distribution parameters for a batch of states.

        Parameters
        ----------
        state:
            Tensor of shape ``(batch, state_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mean, log_std)`` each of shape ``(batch, action_dim)``.
            ``log_std`` is clamped to ``[LOG_STD_MIN, LOG_STD_MAX]``.
        """
        h = self.backbone(state)
        mean = self.mean(h)
        # Clamp log_std to keep std = exp(log_std) numerically stable:
        # too negative → std → 0 (no exploration); too positive → NaN.
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a (differentiable) action sample and compute its log-probability.

        **Reparameterisation trick:** ``z = mean + std * ε`` where ``ε ~ N(0, I)``.
        Unlike ``sample()``, ``rsample()`` propagates gradients through ``mean`` and
        ``std``, which is required for the actor loss.

        **Tanh squashing:** ``squashed = tanh(z)`` bounds the raw sample to ``(-1, 1)``
        before rescaling by ``max_action``.

        **Log-prob correction:** The tanh transform changes the probability density.
        The correct log-prob under the squashed distribution is:

            log π_squashed(a|s) = log π_Gaussian(z|s) − Σ log(1 − tanh²(zᵢ) + ε)

        where the sum runs over action dimensions and ``ε = 1e-6`` prevents ``log(0)``.
        This is the change-of-variables Jacobian for the element-wise ``tanh``.

        Parameters
        ----------
        state:
            Tensor of shape ``(batch, state_dim)``.
        deterministic:
            If ``True``, return the mode of the distribution (``z = mean``, no noise)
            with a dummy zero log-prob.  Used for evaluation and video recording only.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(action, log_prob)`` where ``action`` has shape ``(batch, action_dim)``
            and ``log_prob`` has shape ``(batch, 1)``.
        """
        mean, log_std = self(state)
        std = log_std.exp()

        if deterministic:
            # Evaluation mode: take the mode (no noise, no gradient through sampling).
            z = mean
        else:
            # Training mode: reparameterised sample — differentiable w.r.t. mean/std.
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()

        squashed = torch.tanh(z)
        action = squashed * self.max_action  # rescale to action space

        if deterministic:
            # Dummy log-prob; never used for gradient computation.
            log_prob = torch.zeros((state.shape[0], 1), device=state.device, dtype=state.dtype)
        else:
            # Log-prob under the *unsquashed* Gaussian
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(z)
            # Jacobian correction for tanh squashing (change of variables)
            correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
            # Sum across action dimensions: log_prob shape (batch, action_dim) → (batch, 1)
            log_prob = (log_prob - correction).sum(dim=1, keepdim=True)

        return action, log_prob


class TwinCritic(nn.Module):
    """Two independent Q-function networks (clipped double-Q for SAC).

    Identical in structure to :class:`~benchmarks.td3.TwinCritic`, but SAC does
    **not** need a ``q1_only`` method.  The SAC actor update uses
    ``min(Q1, Q2)`` directly in the actor loss, unlike TD3 which separates the
    actor-driving head from the target-computation head.

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
        """Return Q-values from both critic heads.

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


def run_sac(config: SACConfig | None = None) -> list[float]:
    """Train a SAC agent and return the per-episode reward history.

    Algorithm outline:

    1. **Warmup** — collect ``warmup_steps`` transitions with random actions.
    2. **Data collection** — sample actions stochastically from the current actor
       (no external exploration noise needed; the policy is inherently stochastic).
    3. **Critic update** — train both Q-heads toward the entropy-augmented Bellman target:

           target = r + γ * not_done * (min(Q1_target, Q2_target)(s', a') − α * log π(a'|s'))

       The ``- α * log π`` term subtracts the entropy cost from the bootstrap value,
       encouraging the policy to remain stochastic and avoid premature convergence.

    4. **Actor update** — maximise expected entropy-augmented Q-value:

           L_actor = (α * log π(a|s) − min(Q1, Q2)(s, a)).mean()

       The first term is the entropy penalty (encouraging high-entropy behaviour) and
       the second is the value signal.  Minimising this loss maximises the objective.

    5. **Soft update** — Polyak update for critic target networks only.
       There is **no actor target** in SAC: the stochastic policy naturally smoothes
       the Q surface, making TD3-style target policy smoothing redundant.

    Parameters
    ----------
    config:
        Hyperparameter configuration.  Defaults to :class:`SACConfig` if ``None``.

    Returns
    -------
    list[float]
        Total undiscounted reward per episode in chronological order.
    """
    cfg = config or SACConfig()
    print("\n--- Starting SAC ---")

    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SAC has no actor_target — only critic targets are maintained.
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

            # --- Action selection ---
            if global_step < cfg.warmup_steps:
                # Warmup: uniform random actions.
                action = env.action_space.sample()
            else:
                # Training: stochastic sample from the actor (no added noise needed).
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

            # --- Gradient updates ---
            if replay.size >= max(cfg.warmup_steps, cfg.batch_size):
                for _ in range(cfg.train_updates_per_step):
                    s, a, ns, r, nd = replay.sample(cfg.batch_size)

                    # ---  Critic target  ---
                    with torch.no_grad():
                        # Sample the next action and its entropy from the current actor.
                        next_a, next_logp = actor.sample(ns, deterministic=False)
                        target_q1, target_q2 = critic_target(ns, next_a)
                        # Entropy-augmented Bellman backup: subtract α * log π from bootstrap.
                        target_q = torch.min(target_q1, target_q2) - cfg.alpha * next_logp
                        target = r + nd * cfg.gamma * target_q

                    # --- Critic update: both heads train toward the same target ---
                    current_q1, current_q2 = critic(s, a)
                    critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    # --- Actor update ---
                    # Re-sample actions under the current policy for gradient computation.
                    pi_action, logp_pi = actor.sample(s, deterministic=False)
                    q1_pi, q2_pi = critic(s, pi_action)
                    # Pessimistic Q-estimate for the actor loss: use min(Q1, Q2).
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    # Actor loss = entropy pressure (α * logp) − value signal (min_q).
                    # Minimising this loss pushes the actor toward high-value, high-entropy actions.
                    actor_loss = (cfg.alpha * logp_pi - min_q_pi).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # --- Critic target soft update (no actor target to update) ---
                    with torch.no_grad():
                        for p, tp in zip(critic.parameters(), critic_target.parameters()):
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
            """Deterministic policy wrapper for video recording (no stochastic sampling)."""
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
