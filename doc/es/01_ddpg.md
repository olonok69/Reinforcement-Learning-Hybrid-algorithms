# DDPG (Deep Deterministic Policy Gradient)

## 1) Objetivo
DDPG resuelve tareas de control continuo donde no es viable enumerar acciones para hacer `argmax_a Q(s,a)`.

Usa:
- actor determinista `mu(s)` para producir acciones directas
- critic `Q(s,a)` para evaluar esas acciones
- replay buffer + target networks para estabilidad off-policy

## 2) Ecuaciones principales
Target del critic:

`y = r + gamma * (1 - done) * Q_target(s_next, mu_target(s_next))`

Loss del critic:

`L_critic = MSE(Q(s,a), y)`

Loss del actor:

`L_actor = -mean(Q(s, mu(s)))`

## 3) Mapeo en el repositorio
- Config y loop de entrenamiento: `../../benchmarks/ddpg.py`
- Replay buffer: `../../benchmarks/replay_buffer.py`
- Runner individual: `../../ddpg_benchmark.py`
- Runner unificado: `../../run_all_comparison.py`

## 4) Defaults importantes
- Entorno: `Pendulum-v1`
- Episodios: `160`
- `gamma=0.99`
- `tau=0.005`
- `batch_size=256`
- `warmup_steps=5000`
- ruido de exploracion: Gaussiano (`exploration_noise=0.1`)

## 5) Fortalezas y riesgos
Fortalezas:
- Algoritmo claro para acciones continuas
- Buena reutilizacion de muestras off-policy

Riesgos:
- Sobreestimacion de Q con critic unico
- Sensible a ruido y learning rates

## 6) Comandos de ejemplo
```bash
uv run python ddpg_benchmark.py --episodes 160 --seed 42
uv run python run_all_comparison.py --methods ddpg --output-dir outputs/ddpg_only
```

## 7) Referencias
- https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-4643c1f71b2e/
- https://spinningup.openai.com/en/latest/algorithms/ddpg.html
- https://medium.com/@amaresh.dm/how-ddpg-deep-deterministic-policy-gradient-algorithms-works-in-reinforcement-learning-117e6a932e68
- https://intellabs.github.io/coach/components/agents/policy_optimization/ddpg.html
- https://www.nature.com/articles/s41598-025-99213-3
