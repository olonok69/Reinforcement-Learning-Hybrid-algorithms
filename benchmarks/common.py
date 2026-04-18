"""common — shared utilities for all benchmark algorithms.

Provides:
* :class:`BenchmarkResult`   — structured output dataclass for a single training run.
* :func:`set_global_seed`    — reproducibility helper that seeds stdlib, NumPy, and PyTorch.
* :func:`moving_average_max` — peak rolling-mean reward over the full episode history.
* :func:`moving_average_last`— rolling-mean reward over the final window (convergence proxy).
* :func:`run_timed`          — wraps a training callable, measures wall-clock time, and
                               returns a :class:`BenchmarkResult`.
* :func:`save_results_json`  — serialise a list of results to JSON.
* :func:`save_results_csv`   — serialise a list of results to CSV.
* :func:`record_policy_video_continuous` — evaluate and record a trained policy to video.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import random
import time
import typing as tt

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    """Structured outcome of one complete training run.

    Collected by :func:`run_timed` and persisted by :func:`save_results_json` /
    :func:`save_results_csv`.  Two reward metrics are recorded to capture both
    peak performance and convergence stability:

    Attributes
    ----------
    algo:
        Algorithm identifier string (e.g. ``"DDPG"``, ``"TD3"``, ``"SAC"``).
    episodes:
        Total number of episodes the training ran for.
    elapsed_sec:
        Wall-clock training time in seconds.
    max_avg_reward_100:
        Maximum value of the 100-episode rolling mean seen at any point during
        training.  Measures peak performance — useful when an algorithm converges
        early but may later destabilise.
    final_avg_reward_100:
        Mean reward over the last 100 episodes.  Measures convergence stability
        — a high value here means the algorithm was still performing well at the
        end of the run.
    """

    algo: str
    episodes: int
    elapsed_sec: float
    max_avg_reward_100: float
    final_avg_reward_100: float

    def to_dict(self) -> dict:
        """Return all fields as a plain :class:`dict` (for JSON/CSV serialisation)."""
        return asdict(self)


def set_global_seed(seed: int) -> None:
    """Set the random seed for the Python stdlib, NumPy, and PyTorch simultaneously.

    Call this once at the start of each run — before creating environments or
    instantiating network weights — to ensure reproducible results across runs
    with the same seed.

    Note: this does **not** seed the Gymnasium environment itself.  Pass the
    same ``seed`` value to ``env.reset(seed=seed)`` if full environment
    reproducibility is also required.

    Parameters
    ----------
    seed:
        Integer seed value.  Conventional choices are ``0``, ``1``, ``42``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average_max(values: tt.Sequence[float], window: int = 100) -> float:
    """Return the maximum value of a sliding-window mean over ``values``.

    Slides a uniform kernel of size ``window`` over the episode reward sequence
    and returns the highest average seen at any point during training.  This
    captures peak performance even if the agent later destabilises.

    If ``len(values) < window`` the mean of all values is returned directly
    (no partial window is discarded).

    Parameters
    ----------
    values:
        Sequence of per-episode total rewards (one float per episode).
    window:
        Rolling window size.  Default ``100`` follows the standard benchmark
        convention for continuous-control tasks.

    Returns
    -------
    float
        Maximum rolling-mean reward, or ``0.0`` for an empty sequence.
    """
    if not values:
        return 0.0
    if len(values) < window:
        return float(np.mean(values))
    arr = np.asarray(values, dtype=np.float32)
    # Uniform kernel: equivalent to a box filter of width ``window``
    kernel = np.ones(window, dtype=np.float32) / window
    conv = np.convolve(arr, kernel, mode="valid")  # shape: (len - window + 1,)
    return float(np.max(conv))


def moving_average_last(values: tt.Sequence[float], window: int = 100) -> float:
    """Return the mean reward over the last ``window`` episodes.

    Used as a convergence proxy: high values here indicate the policy was still
    performing well at the end of training.

    If ``len(values) < window`` the mean of all values is returned directly.

    Parameters
    ----------
    values:
        Sequence of per-episode total rewards.
    window:
        Number of trailing episodes to average.  Default ``100``.

    Returns
    -------
    float
        Mean of the last ``window`` rewards, or ``0.0`` for an empty sequence.
    """
    if not values:
        return 0.0
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


def run_timed(train_fn: tt.Callable[[], tt.Sequence[float]], algo: str) -> tuple[tt.Sequence[float], BenchmarkResult]:
    """Execute a training callable, measure wall-clock time, and return a result summary.

    Parameters
    ----------
    train_fn:
        Zero-argument callable that runs the full training loop and returns the
        per-episode reward sequence (e.g. ``lambda: run_ddpg(cfg)``).
    algo:
        Identifier string stored in the resulting :class:`BenchmarkResult`
        (e.g. ``"DDPG"``).

    Returns
    -------
    tuple[list[float], BenchmarkResult]
        * ``rewards``  — the full per-episode reward history returned by ``train_fn``.
        * ``result``   — a :class:`BenchmarkResult` populated with timing and reward metrics.
    """
    start = time.time()
    rewards = list(train_fn())
    elapsed = time.time() - start
    result = BenchmarkResult(
        algo=algo,
        episodes=len(rewards),
        elapsed_sec=elapsed,
        max_avg_reward_100=moving_average_max(rewards, window=100),
        final_avg_reward_100=moving_average_last(rewards, window=100),
    )
    return rewards, result


def save_results_json(results: tt.Sequence[BenchmarkResult], output_path: str | Path) -> None:
    """Serialise a list of benchmark results to a pretty-printed JSON file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    results:
        Sequence of :class:`BenchmarkResult` objects to persist.
    output_path:
        Destination file path (e.g. ``"outputs/comparison_results.json"``).
    """
    payload = [r.to_dict() for r in results]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_results_csv(results: tt.Sequence[BenchmarkResult], output_path: str | Path) -> None:
    """Serialise a list of benchmark results to a CSV file with a header row.

    Parent directories are created automatically if they do not exist.
    The column order matches the field order of :class:`BenchmarkResult`.

    Parameters
    ----------
    results:
        Sequence of :class:`BenchmarkResult` objects to persist.
    output_path:
        Destination file path (e.g. ``"outputs/comparison_results.csv"``).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "episodes",
                "elapsed_sec",
                "max_avg_reward_100",
                "final_avg_reward_100",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())


def record_policy_video_continuous(
    env_name: str,
    video_dir: str,
    episodes: int,
    name_prefix: str,
    policy_fn: tt.Callable[[np.ndarray], np.ndarray],
) -> None:
    """Evaluate a trained policy and save the episodes to video files.

    Wraps the environment with Gymnasium's :class:`~gymnasium.wrappers.RecordVideo`
    and runs ``episodes`` full evaluation episodes using ``policy_fn``.  Videos are
    written to ``video_dir`` with filenames prefixed by ``name_prefix``.

    Requires ``moviepy`` to be installed (``uv pip install moviepy``).  If it is
    missing the function prints a warning and returns without raising.

    Parameters
    ----------
    env_name:
        Gymnasium environment ID (e.g. ``"Pendulum-v1"``).
    video_dir:
        Directory where video files will be written.  Created automatically.
    episodes:
        Number of evaluation episodes to record.
    name_prefix:
        Filename prefix for the recorded video files (e.g. ``"ddpg"``).
    policy_fn:
        Callable that maps a state ``np.ndarray`` to an action ``np.ndarray``.
        Should run in deterministic mode (no exploration noise or sampling).
    """
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    Path(video_dir).mkdir(parents=True, exist_ok=True)
    # render_mode="rgb_array" is required for RecordVideo to capture frames.
    base_env = gym.make(env_name, render_mode="rgb_array")
    try:
        video_env = RecordVideo(
            env=base_env,
            video_folder=video_dir,
            episode_trigger=lambda ep_idx: ep_idx < episodes,  # record all episodes
            name_prefix=name_prefix,
        )
    except Exception as exc:
        # RecordVideo raises if moviepy is absent; degrade gracefully.
        base_env.close()
        print(f"  Video recording unavailable: {exc}")
        print("  Install dependency with: uv pip install moviepy")
        return

    try:
        for i in range(episodes):
            state, _ = video_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = np.asarray(policy_fn(np.asarray(state, dtype=np.float32)), dtype=np.float32)
                state, reward, terminated, truncated, _ = video_env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)
            print(f"  Video episode {i + 1}/{episodes}: reward = {ep_reward:.1f}")
    finally:
        video_env.close()  # always close to flush the video file

    print(f"  Saved {episodes} video(s) to '{video_dir}'")
