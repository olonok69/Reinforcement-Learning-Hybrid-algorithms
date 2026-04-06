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
    algo: str
    episodes: int
    elapsed_sec: float
    max_avg_reward_100: float
    final_avg_reward_100: float

    def to_dict(self) -> dict:
        return asdict(self)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average_max(values: tt.Sequence[float], window: int = 100) -> float:
    if not values:
        return 0.0
    if len(values) < window:
        return float(np.mean(values))
    arr = np.asarray(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    conv = np.convolve(arr, kernel, mode="valid")
    return float(np.max(conv))


def moving_average_last(values: tt.Sequence[float], window: int = 100) -> float:
    if not values:
        return 0.0
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


def run_timed(train_fn: tt.Callable[[], tt.Sequence[float]], algo: str) -> tuple[tt.Sequence[float], BenchmarkResult]:
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
    payload = [r.to_dict() for r in results]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_results_csv(results: tt.Sequence[BenchmarkResult], output_path: str | Path) -> None:
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
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    Path(video_dir).mkdir(parents=True, exist_ok=True)
    base_env = gym.make(env_name, render_mode="rgb_array")
    try:
        video_env = RecordVideo(
            env=base_env,
            video_folder=video_dir,
            episode_trigger=lambda ep_idx: ep_idx < episodes,
            name_prefix=name_prefix,
        )
    except Exception as exc:
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
        video_env.close()

    print(f"  Saved {episodes} video(s) to '{video_dir}'")
