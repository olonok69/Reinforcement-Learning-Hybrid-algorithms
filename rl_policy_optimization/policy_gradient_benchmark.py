import argparse

from benchmarks.common import run_timed, set_global_seed
from benchmarks.policy_gradient import PolicyGradientConfig, run_policy_gradient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Policy Gradient (REINFORCE) benchmark on CartPole-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=700)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--no-normalize-returns",
        action="store_true",
        help="Disable return normalization before policy updates",
    )
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos after training")
    parser.add_argument("--video-dir", default="videos/policy_gradient", help="Directory for recorded videos")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)

    config = PolicyGradientConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        normalize_returns=not args.no_normalize_returns,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
    )
    _, result = run_timed(lambda: run_policy_gradient(config), "Policy Gradient")
    print("\n--- Summary ---")
    print(f"{result.algo} took {result.elapsed_sec:.2f}s for {result.episodes} episodes.")
    print(f"{result.algo} Max Avg Reward (100): {result.max_avg_reward_100:.2f}")
    print(f"{result.algo} Final Avg Reward (100): {result.final_avg_reward_100:.2f}")
