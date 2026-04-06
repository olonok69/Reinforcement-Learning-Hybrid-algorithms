# SAC (Soft Actor-Critic)

## 1) Goal
SAC combines off-policy replay efficiency with stochastic exploration through entropy regularization.

Objective intuition:
- maximize return
- maximize policy entropy (controlled by `alpha`)

## 2) Core target and actor objective
Critic target:

`y = r + gamma * (1 - done) * (min(Q1_target, Q2_target) - alpha * log_pi(a_next|s_next))`

Actor objective:

`L_actor = mean(alpha * log_pi(a|s) - min(Q1(s,a), Q2(s,a)))`

The implementation uses tanh-squashed Gaussian actions with log-probability correction.

## 3) Repository mapping
- Implementation: `../benchmarks/sac.py`
- Replay buffer: `../benchmarks/replay_buffer.py`
- Individual runner: `../sac_benchmark.py`
- Unified benchmark: `../run_all_comparison.py`

## 4) Important defaults
- Environment: `Pendulum-v1`
- Episodes: `160`
- `gamma=0.99`
- `tau=0.005`
- `alpha=0.2` (fixed)
- `batch_size=256`
- `warmup_steps=5000`

## 5) Strengths and risks
Strengths:
- Usually robust and strong default for continuous control
- Built-in exploration via stochastic policy and entropy term

Risks:
- More compute per update than DDPG
- Alpha tuning (or auto-tuning) can materially change behavior

## 6) Example commands
```bash
uv run python sac_benchmark.py --episodes 160 --seed 42
uv run python run_all_comparison.py --methods sac --output-dir outputs/sac_only
```

## 7) References
- https://towardsdatascience.com/navigating-soft-actor-critic-reinforcement-learning-8e1a7406ce48/
- https://spinningup.openai.com/en/latest/algorithms/sac.html
