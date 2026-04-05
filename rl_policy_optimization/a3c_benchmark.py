import argparse

from benchmarks.a3c import A3CConfig, run_a3c
from benchmarks.common import run_timed, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A3C benchmark on CartPole-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos after training")
    parser.add_argument("--video-dir", default="videos/a3c", help="Directory for recorded videos")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)

    config = A3CConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        workers=args.workers,
        rollout_steps=args.rollout_steps,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
    )
    _, result = run_timed(lambda: run_a3c(config), "A3C")
    print("\n--- Summary ---")
    print(f"{result.algo} took {result.elapsed_sec:.2f}s for {result.episodes} episodes.")
    print(f"{result.algo} Max Avg Reward (100): {result.max_avg_reward_100:.2f}")
    print(f"{result.algo} Final Avg Reward (100): {result.final_avg_reward_100:.2f}")
