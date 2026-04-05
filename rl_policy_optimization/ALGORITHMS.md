# Reinforcement Learning: Policy Optimization Methods

This guide explains the reinforcement learning algorithms implemented in this repository — all based on **policy optimization** for `CartPole-v1`.

---

## Core objective of policy optimization

Instead of learning only action-values, policy optimization methods directly optimize a stochastic policy:

$$
\pi_\theta(a|s)
$$

to maximize expected return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

A common policy-gradient estimator is:

$$
\nabla_\theta J(\theta) \propto \mathbb{E}[\nabla_\theta\log\pi_\theta(a_t|s_t)\,\hat{A}_t]
$$

where $\hat{A}_t$ is an advantage estimate that tells whether an action was better or worse than expected.

---

## 1. Policy Gradient (REINFORCE)

REINFORCE is the pure Monte Carlo policy-gradient baseline.

### How it works
1. Roll out one full episode with current policy.
2. Compute discounted returns for each timestep.
3. Increase probability of actions followed by high return.

### Practical characteristics
- Very simple and educational.
- High variance updates.
- Usually less stable than actor-critic variants.

- Main benchmark module: `benchmarks/policy_gradient.py`
- Standalone runner: `policy_gradient_benchmark.py`

---

## 2. A2C (Advantage Actor-Critic)

A2C combines:
- **Actor**: policy network that selects actions
- **Critic**: value estimator used to reduce gradient variance

The advantage term $\hat{A}_t = R_t - V(s_t)$ stabilizes updates compared with plain REINFORCE.

### How it works
1. Collect trajectory data with current policy.
2. Estimate returns and state values.
3. Update actor from advantages; update critic from value loss.

### Practical characteristics
- Strong baseline for on-policy RL.
- More stable than REINFORCE due to critic baseline.
- Adds complexity (joint policy + value optimization).

- Main benchmark module: `benchmarks/a2c.py`
- Standalone runner: `a2c_benchmark.py`

---

## 3. A3C (Asynchronous Advantage Actor-Critic)

A3C extends actor-critic with multiple workers collecting rollouts asynchronously and updating shared parameters.

### How it works
1. Spawn multiple environment workers.
2. Workers collect short rollouts in parallel.
3. Central learner applies gradients and refreshes shared parameters.

### Practical characteristics
- Better data decorrelation than single-worker actor-critic.
- Can improve wall-clock learning speed with CPU parallelism.
- Multiprocessing increases engineering/runtime complexity.

- Main benchmark module: `benchmarks/a3c.py`
- Standalone runner: `a3c_benchmark.py`

---

## 4. PPO (Proximal Policy Optimization)

PPO improves training stability using a clipped surrogate objective to limit large policy updates:

$$
L^{clip}(\theta)=\mathbb{E}[\min(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)]
$$

### How it works
1. Collect rollouts with current policy.
2. Estimate advantages (often with GAE).
3. Run several minibatch epochs on clipped objective.

### Practical characteristics
- Excellent stability vs implementation complexity trade-off.
- Common default in practical policy-optimization pipelines.
- Still on-policy, so data is typically discarded after updates.

- Main benchmark module: `benchmarks/ppo.py`
- Standalone runner: `ppo_benchmark.py`

---

## 5. TRPO (Trust Region Policy Optimization)

TRPO performs constrained policy updates inside a trust region for robust improvement steps.

### How it works
1. Build a local policy-improvement objective.
2. Constrain policy shift with a KL-divergence trust region.
3. Solve constrained step approximately (implemented here through `sb3-contrib`).

### Practical characteristics
- Theoretically principled conservative updates.
- Often very stable.
- Heavier optimization machinery than PPO in practice.

- Main benchmark module: `benchmarks/trpo.py`
- Standalone runner: `trpo_benchmark.py`

---

## Comparison Summary

| Feature | REINFORCE | A2C | A3C | PPO | TRPO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Policy update control** | Unconstrained | Unconstrained | Unconstrained (async data) | Clipped objective | KL trust region |
| **Variance reduction** | Low | Medium-High | Medium-High | High (GAE + clipping) | High |
| **Parallelism support** | Single worker | Single worker | Multi-worker async | Typically single learner | Library-managed |
| **Implementation complexity** | Low | Medium | Medium-High | Medium | High |
| **Typical usage** | Educational baseline | Stable actor-critic baseline | CPU-parallel actor-critic | Practical default baseline | Advanced conservative baseline |
