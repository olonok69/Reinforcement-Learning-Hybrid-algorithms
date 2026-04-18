import argparse

from benchmarks.common import run_timed, set_global_seed
from benchmarks.td3 import TD3Config, run_td3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TD3 benchmark on Pendulum-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-name", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--train-updates-per-step", type=int, default=1)
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos after training")
    parser.add_argument("--video-dir", default="videos/td3", help="Directory for recorded videos")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)

    config = TD3Config(
        env_name=args.env_name,
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        warmup_steps=args.warmup_steps,
        exploration_noise=args.exploration_noise,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        train_updates_per_step=args.train_updates_per_step,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
    )
    _, result = run_timed(lambda: run_td3(config), "TD3")
    print("\n--- Summary ---")
    print(f"{result.algo} took {result.elapsed_sec:.2f}s for {result.episodes} episodes.")
    print(f"{result.algo} Max Avg Reward (100): {result.max_avg_reward_100:.2f}")
    print(f"{result.algo} Final Avg Reward (100): {result.final_avg_reward_100:.2f}")
