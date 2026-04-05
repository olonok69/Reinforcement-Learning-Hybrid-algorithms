import argparse
import json
from pathlib import Path
import traceback
import typing as tt

from benchmarks.common import BenchmarkResult, run_timed, save_results_csv, save_results_json, set_global_seed
from benchmarks.policy_gradient import PolicyGradientConfig, run_policy_gradient
from benchmarks.a2c import A2CConfig, run_a2c
from benchmarks.a3c import A3CConfig, run_a3c
from benchmarks.ppo import PPOConfig, run_ppo
from benchmarks.trpo import TRPOConfig, run_trpo


Runner = tt.Callable[[], list[float]]


def _build_methods(
    policy_gradient_episodes: int = 700,
    a2c_episodes: int = 500,
    a3c_episodes: int = 500,
    ppo_episodes: int = 500,
    trpo_timesteps: int = 100_000,
    trpo_backend: str = "sb3",
    trpo_batch_size: int = 2_048,
    trpo_gamma: float = 0.99,
    trpo_gae_lambda: float = 0.97,
    trpo_max_kl: float = 1e-2,
    trpo_damping: float = 1e-1,
    trpo_cg_steps: int = 10,
    trpo_value_lr: float = 1e-3,
    trpo_value_iters: int = 40,
    trpo_hidden_size: int = 128,
    a3c_workers: int = 4,
    a3c_rollout_steps: int = 5,
    record_video: bool = False,
    video_dir: str = "videos",
    video_episodes: int = 3,
) -> dict[str, tuple[str, Runner]]:
    pg_cfg = PolicyGradientConfig(
        episodes=policy_gradient_episodes,
        record_video=record_video,
        video_dir=f"{video_dir}/policy_gradient",
        video_episodes=video_episodes,
    )
    a2c_cfg = A2CConfig(
        episodes=a2c_episodes,
        record_video=record_video,
        video_dir=f"{video_dir}/a2c",
        video_episodes=video_episodes,
    )
    a3c_cfg = A3CConfig(
        episodes=a3c_episodes,
        workers=a3c_workers,
        rollout_steps=a3c_rollout_steps,
        record_video=record_video,
        video_dir=f"{video_dir}/a3c",
        video_episodes=video_episodes,
    )
    ppo_cfg = PPOConfig(
        episodes=ppo_episodes,
        record_video=record_video,
        video_dir=f"{video_dir}/ppo",
        video_episodes=video_episodes,
    )
    trpo_cfg = TRPOConfig(
        total_timesteps=trpo_timesteps,
        backend=trpo_backend,
        batch_size=trpo_batch_size,
        gamma=trpo_gamma,
        gae_lambda=trpo_gae_lambda,
        max_kl=trpo_max_kl,
        damping=trpo_damping,
        cg_steps=trpo_cg_steps,
        value_lr=trpo_value_lr,
        value_iters=trpo_value_iters,
        hidden_size=trpo_hidden_size,
        record_video=record_video,
        video_dir=f"{video_dir}/trpo",
        video_episodes=video_episodes,
    )

    return {
        "policy_gradient": ("Policy Gradient", lambda: run_policy_gradient(pg_cfg)),
        "a2c": ("A2C", lambda: run_a2c(a2c_cfg)),
        "a3c": ("A3C", lambda: run_a3c(a3c_cfg)),
        "ppo": ("PPO", lambda: run_ppo(ppo_cfg)),
        "trpo": ("TRPO", lambda: run_trpo(trpo_cfg)),
    }


ALL_METHOD_KEYS = ["policy_gradient", "a2c", "a3c", "ppo", "trpo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified RL benchmark comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHOD_KEYS,
        help="Methods to run. Default: all methods.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on first failure instead of continuing with remaining methods.",
    )
    parser.add_argument("--policy-gradient-episodes", type=int, default=700, help="Training episodes for Policy Gradient")
    parser.add_argument("--a2c-episodes", type=int, default=500, help="Training episodes for A2C")
    parser.add_argument("--a3c-episodes", type=int, default=500, help="Training episodes for A3C")
    parser.add_argument("--ppo-episodes", type=int, default=500, help="Training episodes for PPO")
    parser.add_argument("--trpo-timesteps", type=int, default=100_000, help="Training timesteps for TRPO")
    parser.add_argument("--trpo-backend", choices=["sb3", "native"], default="sb3", help="TRPO backend implementation")
    parser.add_argument("--trpo-batch-size", type=int, default=2_048, help="Native TRPO rollout batch size")
    parser.add_argument("--trpo-gamma", type=float, default=0.99, help="Native TRPO discount factor")
    parser.add_argument("--trpo-gae-lambda", type=float, default=0.97, help="Native TRPO GAE lambda")
    parser.add_argument("--trpo-max-kl", type=float, default=1e-2, help="Native TRPO KL constraint")
    parser.add_argument("--trpo-damping", type=float, default=1e-1, help="Native TRPO damping for Fisher-vector product")
    parser.add_argument("--trpo-cg-steps", type=int, default=10, help="Native TRPO conjugate-gradient iterations")
    parser.add_argument("--trpo-value-lr", type=float, default=1e-3, help="Native TRPO value network learning rate")
    parser.add_argument("--trpo-value-iters", type=int, default=40, help="Native TRPO value network updates per batch")
    parser.add_argument("--trpo-hidden-size", type=int, default=128, help="Native TRPO hidden layer width")
    parser.add_argument("--a3c-workers", type=int, default=4, help="Number of parallel workers for A3C")
    parser.add_argument("--a3c-rollout-steps", type=int, default=5, help="Rollout steps per A3C worker update")
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos for selected methods")
    parser.add_argument("--video-dir", default="videos", help="Base directory for recorded videos")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    return parser.parse_args()


def validate_methods(methods: list[str]) -> list[str]:
    unknown = [name for name in methods if name not in ALL_METHOD_KEYS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {ALL_METHOD_KEYS}")
    return methods


def print_summary(results: list[BenchmarkResult], errors: list[dict]) -> None:
    print("\n--- Unified Summary ---")
    if results:
        for r in sorted(results, key=lambda x: x.max_avg_reward_100, reverse=True):
            print(
                f"{r.algo:16} | episodes={r.episodes:4d} | "
                f"max100={r.max_avg_reward_100:7.2f} | final100={r.final_avg_reward_100:7.2f} | "
                f"time={r.elapsed_sec:8.2f}s"
            )
    else:
        print("No successful runs.")

    if errors:
        print("\nFailed methods:")
        for err in errors:
            print(f"- {err['method']}: {err['error']}")


def main() -> None:
    args = parse_args()
    selected_methods = validate_methods(args.methods)
    set_global_seed(args.seed)

    methods = _build_methods(
        policy_gradient_episodes=args.policy_gradient_episodes,
        a2c_episodes=args.a2c_episodes,
        a3c_episodes=args.a3c_episodes,
        ppo_episodes=args.ppo_episodes,
        trpo_timesteps=args.trpo_timesteps,
        trpo_backend=args.trpo_backend,
        trpo_batch_size=args.trpo_batch_size,
        trpo_gamma=args.trpo_gamma,
        trpo_gae_lambda=args.trpo_gae_lambda,
        trpo_max_kl=args.trpo_max_kl,
        trpo_damping=args.trpo_damping,
        trpo_cg_steps=args.trpo_cg_steps,
        trpo_value_lr=args.trpo_value_lr,
        trpo_value_iters=args.trpo_value_iters,
        trpo_hidden_size=args.trpo_hidden_size,
        a3c_workers=args.a3c_workers,
        a3c_rollout_steps=args.a3c_rollout_steps,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
    )

    results: list[BenchmarkResult] = []
    errors: list[dict] = []

    for method_key in selected_methods:
        display_name, runner = methods[method_key]
        print(f"\n=== Running {display_name} ({method_key}) ===")
        try:
            _, result = run_timed(runner, display_name)
            results.append(result)
        except Exception as exc:
            errors.append(
                {
                    "method": method_key,
                    "display_name": display_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERROR] {display_name} failed: {exc}")
            if args.strict:
                break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "comparison_results.json"
    csv_path = output_dir / "comparison_results.csv"
    errors_path = output_dir / "comparison_errors.json"

    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    errors_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")

    print_summary(results, errors)
    print(f"\nSaved results JSON: {json_path}")
    print(f"Saved results CSV:  {csv_path}")
    print(f"Saved errors JSON:  {errors_path}")


if __name__ == "__main__":
    main()
