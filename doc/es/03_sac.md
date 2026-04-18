# SAC (Soft Actor-Critic)

## 1) Objetivo
SAC combina eficiencia off-policy con exploracion estocastica via regularizacion por entropia.

Intuicion del objetivo:
- maximizar retorno
- maximizar entropia de la policy (controlada por `alpha`)

## 2) Target y objetivo del actor
Target del critic:

`y = r + gamma * (1 - done) * (min(Q1_target, Q2_target) - alpha * log_pi(a_next|s_next))`

Objetivo del actor:

`L_actor = mean(alpha * log_pi(a|s) - min(Q1(s,a), Q2(s,a)))`

La implementacion usa acciones Gaussianas con `tanh` y correccion de log-probabilidad.

## 3) Mapeo en el repositorio
- Implementacion: `../../benchmarks/sac.py`
- Replay buffer: `../../benchmarks/replay_buffer.py`
- Runner individual: `../../sac_benchmark.py`
- Runner unificado: `../../run_all_comparison.py`

## 4) Defaults importantes
- Entorno: `Pendulum-v1`
- Episodios: `160`
- `gamma=0.99`
- `tau=0.005`
- `alpha=0.2` (fijo)
- `batch_size=256`
- `warmup_steps=5000`

## 5) Fortalezas y riesgos
Fortalezas:
- Suele ser un default robusto para control continuo
- Exploracion incorporada por policy estocastica + entropia

Riesgos:
- Mas costo computacional por update
- El valor de alpha impacta de forma relevante

## 6) Comandos de ejemplo
```bash
uv run python sac_benchmark.py --episodes 160 --seed 42
uv run python run_all_comparison.py --methods sac --output-dir outputs/sac_only
```

## 7) Referencias
- https://towardsdatascience.com/navigating-soft-actor-critic-reinforcement-learning-8e1a7406ce48/
- https://spinningup.openai.com/en/latest/algorithms/sac.html
