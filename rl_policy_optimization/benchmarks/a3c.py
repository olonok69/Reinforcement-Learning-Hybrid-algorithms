from __future__ import annotations
"""Asynchronous Advantage Actor-Critic (A3C) benchmark on CartPole-v1."""

from dataclasses import dataclass
import multiprocessing as mp
import queue
import traceback
import typing as tt

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.common import record_policy_video


@dataclass
class A3CConfig:
    """Configuration for asynchronous actor-critic training."""

    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    learning_rate: float = 1e-3
    episodes: int = 500
    hidden_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    workers: int = 4
    rollout_steps: int = 5
    record_video: bool = False
    video_dir: str = "videos/a3c"
    video_episodes: int = 3


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic model used by learner and workers."""

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 128):
        # Initialize nn.Module internals.
        super().__init__()
        # Shared feature extractor consumed by both actor and critic heads.
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        # Actor head outputs logits over discrete actions.
        self.policy_head = nn.Linear(hidden_size, n_actions)
        # Critic head predicts scalar state value V(s).
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Build latent features from state input.
        feats = self.shared(x)
        # Return both actor logits and critic value estimates.
        return self.policy_head(feats), self.value_head(feats)


def _snapshot_state_dict(shared_state: tt.Any) -> dict[str, torch.Tensor]:
    """Convert a manager-backed mapping into a regular tensor state dict."""

    # Manager dictionaries are proxy objects. Create a plain dict so
    # load_state_dict can consume normal tensor mapping safely.
    return {k: v for k, v in shared_state.items()}


def _worker_loop(
    worker_id: int,
    cfg: A3CConfig,
    state_dict: dict,
    data_queue: tt.Any,
    error_queue: tt.Any,
    stop_event: tt.Any,
) -> None:
    """Collect async rollouts and push episodes/batches to learner queues."""

    # Keep env reference outside try block for safe cleanup in finally.
    env = None
    try:
        # Each worker owns an independent environment instance.
        env = gym.make(cfg.env_name)
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Local worker copy of actor-critic model.
        net = ActorCritic(obs_size, n_actions, hidden_size=cfg.hidden_size)
        # Initialize worker parameters from shared learner weights.
        net.load_state_dict(_snapshot_state_dict(state_dict))

        # Deterministic per-worker seed offset for reproducibility.
        state, _ = env.reset(seed=1000 + worker_id)
        episode_reward = 0.0

        # Rollout buffers accumulated until rollout_steps is reached.
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []

        # Continue collecting until stop signal is set by learner.
        while not stop_event.is_set():
            # Convert state to model input.
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # Worker acts with current local snapshot of policy.
                logits, _ = net(state_t)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())

            # Environment transition.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append transition pieces to rollout buffers.
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            dones.append(done)
            episode_reward += float(reward)
            state = next_state

            if done:
                try:
                    # Send completed episode reward to learner for metrics.
                    data_queue.put(("episode", episode_reward), timeout=0.1)
                except queue.Full:
                    # Drop metric silently if queue is saturated.
                    pass
                # Start next episode immediately.
                state, _ = env.reset()
                episode_reward = 0.0

            if len(states) >= cfg.rollout_steps:
                # Bootstrap value for unfinished rollout tail.
                next_state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, next_value = net(next_state_t)
                bootstrap = float(next_value.item())
                # No bootstrap if last transition already ended episode.
                if dones[-1]:
                    bootstrap = 0.0

                # Compute discounted returns backward over rollout chunk.
                returns: list[float] = []
                running = bootstrap
                for reward_i, done_i in zip(reversed(rewards), reversed(dones)):
                    # Reset running return across episode boundaries.
                    if done_i:
                        running = 0.0
                    running = reward_i + cfg.gamma * running
                    returns.append(running)
                returns.reverse()

                # Serialize rollout chunk to numpy arrays for efficient IPC.
                payload = {
                    "states": np.asarray(states, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.int64),
                    "returns": np.asarray(returns, dtype=np.float32),
                }
                try:
                    # Send training batch to learner.
                    data_queue.put(("batch", payload), timeout=0.1)
                except queue.Full:
                    # If queue is full, drop this chunk to keep worker responsive.
                    pass

                # Clear rollout buffers for next chunk.
                states.clear()
                actions.clear()
                rewards.clear()
                dones.clear()

            if stop_event.is_set():
                break

            # Refresh local worker weights from shared learner parameters.
            net.load_state_dict(_snapshot_state_dict(state_dict))
    except Exception as exc:
        # Forward worker exception details to learner.
        try:
            error_queue.put(
                {
                    "worker_id": worker_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                timeout=0.1,
            )
        except queue.Full:
            # If even error queue is full, fail silently to avoid deadlock.
            pass
    finally:
        # Ensure environment process resources are closed.
        if env is not None:
            env.close()


def run_a3c(config: A3CConfig | None = None) -> list[float]:
    """Train an A3C agent and return collected episode rewards."""

    # Use provided configuration or defaults.
    cfg = config or A3CConfig()
    print("\n--- Starting A3C ---")

    # `spawn` is safer/more portable for PyTorch + multiprocessing on Windows.
    mp_ctx = mp.get_context("spawn")

    # Probe env once to get state/action dimensions.
    probe_env = gym.make(cfg.env_name)
    obs_size = probe_env.observation_space.shape[0]
    n_actions = probe_env.action_space.n
    probe_env.close()

    # Central learner model and optimizer.
    net = ActorCritic(obs_size, n_actions, hidden_size=cfg.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    # Shared objects for inter-process communication.
    manager = mp_ctx.Manager()
    # Store parameters in manager dict so workers can pull latest weights.
    shared_state = manager.dict({k: v.cpu() for k, v in net.state_dict().items()})
    # Queue for episode metrics + training batches from workers.
    data_queue = mp_ctx.Queue(maxsize=1024)
    # Queue for worker exceptions so learner can fail fast.
    error_queue = mp_ctx.Queue(maxsize=max(16, cfg.workers * 4))
    # Broadcast stop signal to all workers.
    stop_event = mp_ctx.Event()

    workers: list[mp.Process] = []
    # Push initial learner parameters to shared state.
    initial_state = {k: v.cpu() for k, v in net.state_dict().items()}
    shared_state.update(initial_state)

    # Spawn worker processes.
    for worker_id in range(cfg.workers):
        p = mp_ctx.Process(
            target=_worker_loop,
            args=(worker_id, cfg, shared_state, data_queue, error_queue, stop_event),
        )
        p.start()
        workers.append(p)

    # Collected episode rewards used as benchmark output.
    rewards: list[float] = []
    # Store worker failures discovered during training.
    worker_errors: list[dict[str, str]] = []
    try:
        # Train until target number of episodes is reached.
        while len(rewards) < cfg.episodes:
            # Drain all pending worker errors without blocking.
            while True:
                try:
                    worker_errors.append(error_queue.get_nowait())
                except queue.Empty:
                    break

            if worker_errors:
                # Surface first worker failure with traceback.
                first = worker_errors[0]
                raise RuntimeError(
                    f"A3C worker {first.get('worker_id')} failed: {first.get('error')}\n"
                    f"{first.get('traceback', '')}"
                )

            try:
                # Receive either episode metric event or training batch event.
                msg_type, payload = data_queue.get(timeout=1.0)
            except queue.Empty:
                # If queue is empty and workers are dead, fail hard.
                if not any(p.is_alive() for p in workers):
                    raise RuntimeError("All A3C workers stopped before reaching target episodes.")
                continue

            if msg_type == "episode":
                # Episode reward event for benchmark logging.
                rewards.append(float(payload))
                if len(rewards) % 50 == 0:
                    avg_reward = float(np.mean(rewards[-50:]))
                    print(f"Episode {len(rewards)}, Average Reward (last 50): {avg_reward:.2f}")
                continue

            if msg_type == "batch":
                # Convert rollout payload back into tensors.
                states = torch.tensor(payload["states"], dtype=torch.float32)
                actions = torch.tensor(payload["actions"], dtype=torch.int64)
                returns_t = torch.tensor(payload["returns"], dtype=torch.float32)

                # Learner forward pass.
                logits, values = net(states)
                values = values.squeeze(-1)

                # Reconstruct action distribution to compute actor objective.
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Advantage estimate and actor/critic losses.
                advantages = returns_t - values.detach()
                policy_loss = -(log_probs * advantages).mean()
                value_loss = nn.functional.mse_loss(values, returns_t)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                # Backprop on learner network.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Publish updated learner weights for worker refresh.
                shared_state.update({k: v.cpu() for k, v in net.state_dict().items()})
    finally:
        # Ensure all workers receive stop signal and terminate cleanly.
        stop_event.set()
        for p in workers:
            p.join(timeout=2.0)
            if p.is_alive():
                # Force-kill stragglers to avoid orphaned processes.
                p.terminate()
                p.join(timeout=1.0)

    # Optional post-training video generation using greedy policy.
    if cfg.record_video:
        net.eval()

        def _policy(state: np.ndarray) -> int:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, _ = net(state_t)
                return int(torch.argmax(logits, dim=1).item())

        record_policy_video(
            env_name=cfg.env_name,
            video_dir=cfg.video_dir,
            episodes=cfg.video_episodes,
            name_prefix="a3c",
            policy_fn=_policy,
        )

    # Return collected episode rewards for benchmark summary.
    return rewards
