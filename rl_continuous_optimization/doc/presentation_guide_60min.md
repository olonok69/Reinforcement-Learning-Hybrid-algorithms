# RL Continuous Control - Presentation Guide (60 minutes)

## Scope
This session covers three off-policy actor-critic methods for continuous action spaces:
- DDPG
- TD3
- SAC

Repository context:
- Benchmark package: `../benchmarks/`
- Individual runners: `../ddpg_benchmark.py`, `../td3_benchmark.py`, `../sac_benchmark.py`
- Unified comparison: `../run_all_comparison.py`

## Learning objectives
By the end of the session, the audience should be able to:
1. Explain why DQN-style argmax does not scale to continuous actions.
2. Describe the actor-critic loop in DDPG.
3. Explain TD3's three stability improvements.
4. Explain SAC's entropy-regularized objective and stochastic policy behavior.
5. Run and compare all three methods in this repository.

## 60-minute flow
- 0-5 min: continuous action problem and DQN limitation.
- 5-20 min: DDPG (actor, critic, replay, target networks).
- 20-35 min: TD3 (twin critics, delayed policy updates, target smoothing).
- 35-50 min: SAC (stochastic policy, entropy term, twin critics).
- 50-57 min: side-by-side comparison and practical recommendations.
- 57-60 min: Q&A.

## Concept-to-code map
- DDPG doc: `01_ddpg.md`
- TD3 doc: `02_td3.md`
- SAC doc: `03_sac.md`

## Demo commands
Run from `rl_continuous_optimization/`.

```bash
# Individual benchmarks
uv run python ddpg_benchmark.py --episodes 160 --seed 42
uv run python td3_benchmark.py --episodes 160 --seed 42
uv run python sac_benchmark.py --episodes 160 --seed 42

# Unified comparison
uv run python run_all_comparison.py --methods ddpg td3 sac --output-dir outputs

# Aggregate and report
uv run python scripts/aggregate_results.py --inputs outputs/*.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Continuous Control Aggregate Report"
```

## Practical recommendation slide
1. Start with SAC as robust default.
2. Compare with TD3 for deterministic-policy stability behavior.
3. Keep DDPG as pedagogical baseline and for ablation.

## References
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
