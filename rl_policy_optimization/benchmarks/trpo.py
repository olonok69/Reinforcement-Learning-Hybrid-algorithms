from __future__ import annotations
"""Trust Region Policy Optimization (TRPO) benchmark wrapper.

This module delegates optimization details to `sb3-contrib` and exposes a
consistent benchmark interface used by the repository orchestrators.
"""

from dataclasses import dataclass
import importlib
from typing import Literal

import gymnasium as gym
import numpy as np

from benchmarks.common import record_policy_video
from benchmarks.trpo_native import TRPONativeConfig, run_trpo_native


@dataclass
class TRPOConfig:
    """Configuration for TRPO benchmark execution."""

    env_name: str = "CartPole-v1"
    total_timesteps: int = 100_000
    backend: Literal["sb3", "native"] = "sb3"
    record_video: bool = False
    video_dir: str = "videos/trpo"
    video_episodes: int = 3

    # Native TRPO-specific parameters (used when backend="native").
    batch_size: int = 2_048
    gamma: float = 0.99
    gae_lambda: float = 0.97
    max_kl: float = 1e-2
    damping: float = 1e-1
    cg_steps: int = 10
    value_lr: float = 1e-3
    value_iters: int = 40
    hidden_size: int = 128


def run_trpo(config: TRPOConfig | None = None) -> list[float]:
    """Train a TRPO agent and return monitored episode rewards."""

    # Use supplied config or default benchmark settings.
    cfg = config or TRPOConfig()

    if cfg.backend == "native":
        native_cfg = TRPONativeConfig(
            env_name=cfg.env_name,
            total_timesteps=cfg.total_timesteps,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            max_kl=cfg.max_kl,
            damping=cfg.damping,
            cg_steps=cfg.cg_steps,
            value_lr=cfg.value_lr,
            value_iters=cfg.value_iters,
            hidden_size=cfg.hidden_size,
            record_video=cfg.record_video,
            video_dir=cfg.video_dir,
            video_episodes=cfg.video_episodes,
        )
        return run_trpo_native(native_cfg)

    print("\n--- Starting TRPO ---")

    try:
        # Import TRPO lazily so users can still run non-TRPO scripts
        # without installing sb3-contrib.
        trpo_module = importlib.import_module("sb3_contrib")
        # Extract TRPO class from imported module.
        TRPO = trpo_module.TRPO
        # Import monitor/callback helpers used to capture per-episode rewards.
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        # Raise a clear actionable message if dependency is missing.
        raise ImportError(
            "TRPO benchmark requires sb3-contrib. Install with: pip install sb3-contrib"
        ) from exc

    # Reward history collected from Monitor episode info dictionaries.
    rewards: list[float] = []

    # Local callback class to hook into SB3 training steps.
    class EpisodeRewardCallback(BaseCallback):
        def _on_step(self) -> bool:
            # `infos` is provided by vectorized environment step outputs.
            infos = self.locals.get("infos", [])
            for info in infos:
                # Monitor injects episode summary into info at episode end.
                ep = info.get("episode")
                if ep is not None and "r" in ep:
                    # Append completed-episode reward for benchmark metrics.
                    rewards.append(float(ep["r"]))
            # Returning True tells SB3 to continue training.
            return True

    # Wrap environment with Monitor so episode stats are emitted in info dicts.
    env = Monitor(gym.make(cfg.env_name))

    # Create TRPO model with MLP policy.
    model = TRPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
    )

    # Instantiate callback and run training for requested timesteps.
    callback = EpisodeRewardCallback()
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

    # Optional post-training deterministic evaluation video.
    if cfg.record_video:
        # SB3 predict() API wrapper for shared video helper.
        def _policy(state: np.ndarray) -> int:
            action, _ = model.predict(state, deterministic=True)
            return int(action)

        # Shared recording helper handles environment wrapping and saving.
        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="trpo",
            policy_fn=_policy,
        )

    # Release environment resources.
    env.close()
    # Return collected episode rewards for benchmark summary.
    return rewards
