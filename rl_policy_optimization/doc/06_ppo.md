# PPO (Proximal Policy Optimization)

## Core idea
PPO constrains policy updates using a clipped surrogate objective to avoid destructive large updates while remaining simple to optimize.

Clipped objective:

$$
L^{CLIP}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)A_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]
$$

where

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

Intuition:
- If the new policy changes action probability too much, clipping truncates the incentive to keep pushing in that direction.
- This creates a soft trust region without solving the full constrained optimization used by TRPO.

## How this repo implements it
- Module: `benchmarks/ppo.py`
- Runner: `ppo_benchmark.py`
- Uses rollout collection + GAE + multiple minibatch epochs.

## Training flow
1. Collect rollout transitions with old policy.
2. Compute advantages using GAE.
3. Normalize advantages.
4. Run several epochs of minibatch updates with clip loss.
5. Jointly optimize policy and value objectives (+ entropy bonus).

PPO is usually sensitive to the interaction among rollout size, minibatch size, and number of update epochs.

## Strengths
- Robust and widely used in practice.
- Better stability than plain policy gradients.
- Reasonable implementation complexity.

## Limitations
- Still on-policy; sample efficiency can lag off-policy methods.
- Sensitive to rollout size, epochs, and clip range.

## Common tuning levers
- **Clip range ($\epsilon$)**: lower is more conservative; higher allows faster but riskier updates.
- **Epochs per rollout**: too many can overfit stale data.
- **GAE $\lambda$**: controls bias-variance trade-off in advantage estimates.
- **Value loss coefficient**: too high can overpower policy improvement.

## Practical notes
- Common failure mode: too many update epochs overfit stale rollout.
- Keep observation/reward preprocessing consistent across methods for fair comparisons.
- If KL divergence spikes, reduce learning rate or clip range.

---

## Repository code anchors (practical mapping)
- Main implementation: [../benchmarks/ppo.py](../benchmarks/ppo.py)
- Standalone runner: [../ppo_benchmark.py](../ppo_benchmark.py)
- Comparison orchestrator: [../run_all_comparison.py](../run_all_comparison.py)

Key symbols to present live:
- `PPOConfig`
- `_compute_gae(...)`
- `run_ppo(...)`

---

## Suggested external sources
- https://huggingface.co/blog/deep-rl-ppo
- https://docs.pytorch.org/rl/0.7/tutorials/multiagent_ppo.html
