# DDPG (Deep Deterministic Policy Gradient)

## 1) Goal
DDPG solves continuous-control tasks where action values are not enumerable, so `argmax_a Q(s,a)` is not practical.

It uses:
- deterministic actor `mu(s)` to output actions directly
- critic `Q(s,a)` to evaluate actor choices
- replay buffer + target networks for stable off-policy learning

## 2) Core update equations
Critic target:

`y = r + gamma * (1 - done) * Q_target(s_next, mu_target(s_next))`

Critic loss:

`L_critic = MSE(Q(s,a), y)`

Actor loss:

`L_actor = -mean(Q(s, mu(s)))`

## 3) Repository mapping
- Config and train loop: `../benchmarks/ddpg.py`
- Replay buffer: `../benchmarks/replay_buffer.py`
- Individual runner: `../ddpg_benchmark.py`
- Unified benchmark: `../run_all_comparison.py`

## 4) Important defaults
- Environment: `Pendulum-v1`
- Episodes: `160`
- `gamma=0.99`
- `tau=0.005`
- `batch_size=256`
- `warmup_steps=5000`
- exploration noise: Gaussian (`exploration_noise=0.1`)

## 5) Strengths and risks
Strengths:
- Conceptually simple actor-critic for continuous actions
- Efficient off-policy replay reuse

Risks:
- Prone to Q overestimation with single critic
- Sensitive to noise and learning-rate choices

## 6) Example commands
```bash
uv run python ddpg_benchmark.py --episodes 160 --seed 42
uv run python run_all_comparison.py --methods ddpg --output-dir outputs/ddpg_only
```

## 7) References
- https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-4643c1f71b2e/
- https://spinningup.openai.com/en/latest/algorithms/ddpg.html
- https://medium.com/@amaresh.dm/how-ddpg-deep-deterministic-policy-gradient-algorithms-works-in-reinforcement-learning-117e6a932e68
- https://intellabs.github.io/coach/components/agents/policy_optimization/ddpg.html
- https://www.nature.com/articles/s41598-025-99213-3
