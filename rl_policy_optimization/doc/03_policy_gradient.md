# Policy Gradient (REINFORCE)

## 1) What REINFORCE is solving
REINFORCE is a policy-gradient method that **optimizes the policy directly** instead of learning a value table first.

Given a stochastic policy $\pi_\theta(a|s)$, we maximize expected return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

The key estimator is:

$$
\nabla_\theta J(\theta) \propto \mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\,G_t\right]
$$

where $G_t$ is the discounted return from time step $t$.

---

## 2) Intuition (step-by-step)
REINFORCE follows a Monte Carlo pattern:
1. Run one full episode with the current stochastic policy.
2. Compute discounted returns for each step.
3. Increase probability of actions that led to high return.
4. Decrease probability of actions that led to low return.
5. Repeat over many episodes.

This is simple and conceptually clean, but high variance is common because full-episode returns are noisy.

---

## 3) Why log-probabilities are used
The update uses $\log\pi_\theta(a_t|s_t)$ (not raw probabilities) because:
- It gives numerically stable gradient expressions.
- It naturally appears from the log-derivative trick in policy gradient derivations.
- It makes trajectory-probability products easier to handle in optimization.

In code, this appears as `dist.log_prob(action)`.

---

## 4) Where variance comes from
REINFORCE uses complete-episode returns:

$$
G_t = r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\dots
$$

Even for similar states, sampled trajectories can vary a lot, so gradient estimates can oscillate.

Common mitigation strategies:
- Return normalization
- Baselines / critics (actor-critic family)
- Entropy regularization (optional) to avoid early deterministic collapse

---

## 5) How this repo implements REINFORCE
There are two relevant implementations:

- Benchmark module: [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
- Demo/CLI script: [Policy_Gradient/reinforce_cartpole.py](../Policy_Gradient/reinforce_cartpole.py)

The script [Policy_Gradient/reinforce_cartpole.py](../Policy_Gradient/reinforce_cartpole.py) is the easiest one to demo and record.

### Code follow map (script)
- Policy network definition: [Policy_Gradient/reinforce_cartpole.py](../Policy_Gradient/reinforce_cartpole.py#L13)
- Discounted return computation + normalization: [Policy_Gradient/reinforce_cartpole.py](../Policy_Gradient/reinforce_cartpole.py#L29)
- Main training loop (sample action, collect trajectory, update): [Policy_Gradient/reinforce_cartpole.py](../Policy_Gradient/reinforce_cartpole.py#L41)
- Human rendering evaluation: [Policy_Gradient/reinforce_cartpole.py](../Policy_Gradient/reinforce_cartpole.py#L111)
- Video recording evaluation: [Policy_Gradient/reinforce_cartpole.py](../Policy_Gradient/reinforce_cartpole.py#L139)
- CLI arguments: [Policy_Gradient/reinforce_cartpole.py](../Policy_Gradient/reinforce_cartpole.py#L180)

### Update in this script
This script computes normalized Monte Carlo returns and applies:

$$
\mathcal{L}_{policy} = -\sum_t \log\pi_\theta(a_t|s_t)\,\hat{G}_t
$$

where $\hat{G}_t$ is normalized return.

---

## 6) CLI commands (tested)
Run from repository root.

### Quick smoke test
```bash
uv run python Policy_Gradient/reinforce_cartpole.py --episodes 5 --log-every 1 --seed 42
```

### Standard training run
```bash
uv run python Policy_Gradient/reinforce_cartpole.py --episodes 1200 --gamma 0.99 --lr 1e-3 --log-every 25 --seed 42
```

### Record demo video after training
```bash
uv run python Policy_Gradient/reinforce_cartpole.py --episodes 300 --record-video --video-episodes 3 --video-dir outputs/videos/reinforce_cartpole_demo --seed 42
```

### Render + record for a live demo
```bash
uv run python Policy_Gradient/reinforce_cartpole.py --episodes 300 --record-and-render --render-episodes 2 --video-episodes 2 --video-dir outputs/videos/reinforce_cartpole_demo --seed 42
```

Output videos are written under the folder given by `--video-dir`.

---

## 7) Demo checklist (for recording)
1. Run a short training with fixed seed.
2. Record at least 1 evaluation episode with `--record-video`.
3. Confirm files exist in `outputs/videos/reinforce_cartpole_demo`.
4. Optionally render with `--render-eval` when presenting live.

---

## 8) Strengths and limitations
### Strengths
- Very clear algorithmic logic.
- Works naturally with stochastic policies.
- Good educational foundation for actor-critic/PPO.

### Limitations
- High variance updates.
- On-policy sample inefficiency.
- Sensitive to learning rate, return scaling, and seed.

---

## 9) Practical tuning tips
- If learning is unstable, reduce `--lr`.
- If convergence is slow/noisy, increase episode count and compare multiple seeds.
- Keep $\gamma$ close to $0.99$ for CartPole unless you have a reason to bias toward immediate reward.
- Use return normalization (already done in script) to stabilize gradients.

---

## 10) Reference used for enrichment
This page was enriched using ideas from the following article (rephrased and tied to this codebase):
- https://shivang-ahd.medium.com/policy-gradient-methods-with-reinforce-a-step-by-step-guide-to-reinforcement-learning-mastery-51fe855a504f
