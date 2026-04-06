import argparse
import json
from pathlib import Path
import traceback
import typing as tt

from benchmarks.common import BenchmarkResult, run_timed, save_results_csv, save_results_json, set_global_seed
from benchmarks.ddpg import DDPGConfig, run_ddpg
from benchmarks.td3 import TD3Config, run_td3
from benchmarks.sac import SACConfig, run_sac


Runner = tt.Callable[[], list[float]]


ALL_METHOD_KEYS = ["ddpg", "td3", "sac"]


def _build_methods(
    env_name: str = "Pendulum-v1",
    ddpg_episodes: int = 160,
    td3_episodes: int = 160,
    sac_episodes: int = 160,
    max_steps: int = 200,
    warmup_steps: int = 5_000,
    batch_size: int = 256,
    record_video: bool = False,
    video_dir: str = "videos",
    video_episodes: int = 3,
) -> dict[str, tuple[str, Runner]]:
    ddpg_cfg = DDPGConfig(
        env_name=env_name,
        episodes=ddpg_episodes,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        record_video=record_video,
        video_dir=f"{video_dir}/ddpg",
        video_episodes=video_episodes,
    )
    td3_cfg = TD3Config(
        env_name=env_name,
        episodes=td3_episodes,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        record_video=record_video,
        video_dir=f"{video_dir}/td3",
        video_episodes=video_episodes,
    )
    sac_cfg = SACConfig(
        env_name=env_name,
        episodes=sac_episodes,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        record_video=record_video,
        video_dir=f"{video_dir}/sac",
        video_episodes=video_episodes,
    )

    return {
        "ddpg": ("DDPG", lambda: run_ddpg(ddpg_cfg)),
        "td3": ("TD3", lambda: run_td3(td3_cfg)),
        "sac": ("SAC", lambda: run_sac(sac_cfg)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified continuous-control RL benchmark comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-name", default="Pendulum-v1")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHOD_KEYS,
        help="Methods to run. Default: all methods.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--ddpg-episodes", type=int, default=160)
    parser.add_argument("--td3-episodes", type=int, default=160)
    parser.add_argument("--sac-episodes", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--strict", action="store_true", help="Stop on first failure.")
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos")
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
                f"{r.algo:8} | episodes={r.episodes:4d} | "
                f"max100={r.max_avg_reward_100:8.2f} | final100={r.final_avg_reward_100:8.2f} | "
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
        env_name=args.env_name,
        ddpg_episodes=args.ddpg_episodes,
        td3_episodes=args.td3_episodes,
        sac_episodes=args.sac_episodes,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
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
