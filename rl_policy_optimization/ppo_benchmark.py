import argparse

from benchmarks.common import run_timed, set_global_seed
from benchmarks.ppo import PPOConfig, run_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO benchmark on CartPole-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos after training")
    parser.add_argument("--video-dir", default="videos/ppo", help="Directory for recorded videos")
    parser.add_argument("--video-episodes", type=int, default=3, help="Number of evaluation episodes to record")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)

    config = PPOConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
    )
    _, result = run_timed(lambda: run_ppo(config), "PPO")
    print("\n--- Summary ---")
    print(f"{result.algo} took {result.elapsed_sec:.2f}s for {result.episodes} episodes.")
    print(f"{result.algo} Max Avg Reward (100): {result.max_avg_reward_100:.2f}")
    print(f"{result.algo} Final Avg Reward (100): {result.final_avg_reward_100:.2f}")
