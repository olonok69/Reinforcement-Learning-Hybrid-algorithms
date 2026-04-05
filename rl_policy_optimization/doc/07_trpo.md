# TRPO (Trust Region Policy Optimization)

## Core idea
TRPO enforces a trust-region constraint on policy updates by limiting KL divergence between old and new policies:

$$
\max_\theta\; \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}A(s,a)\right]
\quad\text{s.t.}\quad
\mathbb{E}[D_{KL}(\pi_{old}\|\pi_\theta)] \leq \delta
$$

Compared with PPO, TRPO solves a more constrained optimization problem (often via conjugate gradient + line search).

Intuition:
- TRPO asks for the best expected improvement that still keeps the new policy close to the old one in distribution space.
- The KL bound $\delta$ acts like a safety budget for each update step.

## How this repo implements it
- Module: `benchmarks/trpo.py`
- Runner: `trpo_benchmark.py`
- Uses `sb3-contrib` TRPO implementation for reliability and maintainability.

## Strengths
- Strongly motivated monotonic-improvement perspective.
- Stable updates with explicit trust-region control.

## Optimization perspective
In practice, TRPO approximates second-order information (through Fisher-vector products) and uses line search to satisfy the KL constraint. This is why it can be more stable per step but heavier computationally than first-order clipped methods.

## Limitations
- More complex optimization internals than PPO.
- Heavier implementation burden when built from scratch.

## Practical notes
- This repository intentionally delegates optimization internals to `sb3-contrib`.
- If dependency is missing, the script reports a clear install message.
- For production workflows, PPO is usually preferred for simplicity and speed.
- If TRPO appears too slow for iterative experimentation, use PPO as the default baseline and reserve TRPO for stricter update-control scenarios.

---

## Repository code anchors (practical mapping)
- Main integration wrapper: [../benchmarks/trpo.py](../benchmarks/trpo.py)
- Standalone runner: [../trpo_benchmark.py](../trpo_benchmark.py)
- Comparison orchestrator: [../run_all_comparison.py](../run_all_comparison.py)

Key symbols to present live:
- `TRPOConfig`
- `EpisodeRewardCallback`
- `run_trpo(...)`

---

## Suggested external sources
- https://dilithjay.com/blog/trpo
- https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9
- https://towardsdatascience.com/trust-region-policy-optimization-trpo-explained-4b56bd206fc2/
