# RL Continuous Control Benchmarks

Benchmark suite for continuous-action, off-policy actor-critic methods:
- DDPG
- TD3
- SAC

This module mirrors the workflow used in `rl_policy_optimization`:
- independent script per algorithm
- unified orchestrator for side-by-side comparison
- post-processing scripts for multi-seed aggregation and report generation

## Structure
- `benchmarks/`: reusable training implementations and shared utilities
- `ddpg_benchmark.py`: run DDPG only
- `td3_benchmark.py`: run TD3 only
- `sac_benchmark.py`: run SAC only
- `run_all_comparison.py`: run selected or all methods in one command
- `scripts/aggregate_results.py`: aggregate multiple comparison outputs
- `scripts/generate_aggregate_report.py`: generate plots and Markdown report

## Quick start
From `rl_continuous_optimization/`:

```bash
uv run python ddpg_benchmark.py
uv run python td3_benchmark.py
uv run python sac_benchmark.py
```

Run all together:

```bash
uv run python run_all_comparison.py
```

Run selected methods:

```bash
uv run python run_all_comparison.py --methods ddpg td3
```

Run with videos:

```bash
uv run python run_all_comparison.py --record-video --video-dir videos --video-episodes 3
```

Outputs are written to:
- `outputs/comparison_results.json`
- `outputs/comparison_results.csv`
- `outputs/comparison_errors.json`

## Multi-seed aggregation
Example across seed runs:

```bash
uv run python scripts/aggregate_results.py --inputs runs/seed1/comparison_results.json runs/seed2/comparison_results.json runs/seed3/comparison_results.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
```

Generate report:

```bash
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Continuous Control Aggregate Report"
```

## Suggested references
### DDPG
- https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-4643c1f71b2e/
- https://spinningup.openai.com/en/latest/algorithms/ddpg.html
- https://medium.com/@amaresh.dm/how-ddpg-deep-deterministic-policy-gradient-algorithms-works-in-reinforcement-learning-117e6a932e68
- https://intellabs.github.io/coach/components/agents/policy_optimization/ddpg.html
- https://www.nature.com/articles/s41598-025-99213-3

### TD3
- https://medium.com/@heyamit10/twin-delayed-deep-deterministic-policy-gradient-td3-fc8e9950f029
- https://spinningup.openai.com/en/latest/algorithms/td3.html
- https://discovery.ucl.ac.uk/id/eprint/10210972/1/A%20TD3-Based%20Reinforcement%20Learning%20Algorithm%20with%20Curriculum%20Learning%20for%20Adaptive%20Yaw%20Control%20in%20All-Wheel-Drive%20Electr.pdf

### SAC
- https://towardsdatascience.com/navigating-soft-actor-critic-reinforcement-learning-8e1a7406ce48/
- https://spinningup.openai.com/en/latest/algorithms/sac.html
