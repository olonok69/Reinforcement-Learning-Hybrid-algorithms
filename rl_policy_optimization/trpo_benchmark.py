import argparse

from benchmarks.common import run_timed, set_global_seed
from benchmarks.trpo import TRPOConfig, run_trpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRPO benchmark on CartPole-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument(
        "--backend",
        choices=["sb3", "native"],
        default="sb3",
        help="TRPO backend implementation",
    )
    parser.add_argument("--batch-size", type=int, default=2_048, help="Native TRPO rollout batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Native TRPO discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.97, help="Native TRPO GAE lambda")
    parser.add_argument("--max-kl", type=float, default=1e-2, help="Native TRPO KL constraint")
    parser.add_argument("--damping", type=float, default=1e-1, help="Native TRPO damping for Fisher-vector product")
    parser.add_argument("--cg-steps", type=int, default=10, help="Native TRPO conjugate-gradient iterations")
    parser.add_argument("--value-lr", type=float, default=1e-3, help="Native TRPO value network learning rate")
    parser.add_argument("--value-iters", type=int, default=40, help="Native TRPO value network updates per batch")
    parser.add_argument("--hidden-size", type=int, default=128, help="Native TRPO hidden layer width")
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos after training")
    parser.add_argument("--video-dir", default="videos/trpo", help="Directory for recorded videos")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)

    config = TRPOConfig(
        total_timesteps=args.timesteps,
        backend=args.backend,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_kl=args.max_kl,
        damping=args.damping,
        cg_steps=args.cg_steps,
        value_lr=args.value_lr,
        value_iters=args.value_iters,
        hidden_size=args.hidden_size,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
    )
    _, result = run_timed(lambda: run_trpo(config), "TRPO")
    print("\n--- Summary ---")
    print(f"{result.algo} took {result.elapsed_sec:.2f}s for {result.episodes} episodes.")
    print(f"{result.algo} Max Avg Reward (100): {result.max_avg_reward_100:.2f}")
    print(f"{result.algo} Final Avg Reward (100): {result.final_avg_reward_100:.2f}")
