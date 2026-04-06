# RL Control Continuo - Guia de Presentacion (60 minutos)

## Alcance
Esta sesion cubre tres metodos actor-critic off-policy para espacios de accion continuos:
- DDPG
- TD3
- SAC

Contexto en el repositorio:
- Paquete de benchmarks: `../benchmarks/`
- Runners individuales: `../ddpg_benchmark.py`, `../td3_benchmark.py`, `../sac_benchmark.py`
- Comparacion unificada: `../run_all_comparison.py`

## Objetivos de aprendizaje
Al terminar la sesion, la audiencia deberia poder:
1. Explicar por que `argmax` estilo DQN no escala en acciones continuas.
2. Describir el loop actor-critic de DDPG.
3. Explicar las tres mejoras de estabilidad de TD3.
4. Explicar la entropia y la policy estocastica en SAC.
5. Ejecutar y comparar los tres metodos en este repositorio.

## Flujo de 60 minutos
- 0-5 min: problema de accion continua y limitacion de DQN.
- 5-20 min: DDPG (actor, critic, replay, target networks).
- 20-35 min: TD3 (twin critics, delayed updates, target smoothing).
- 35-50 min: SAC (policy estocastica, termino de entropia, twin critics).
- 50-57 min: comparacion lado a lado y recomendaciones practicas.
- 57-60 min: Q&A.

## Mapa concepto-codigo
- Documento DDPG: `es/01_ddpg.md`
- Documento TD3: `es/02_td3.md`
- Documento SAC: `es/03_sac.md`

## Comandos de demo
Ejecutar desde `rl_continuous_optimization/`.

```bash
# Benchmarks individuales
uv run python ddpg_benchmark.py --episodes 160 --seed 42
uv run python td3_benchmark.py --episodes 160 --seed 42
uv run python sac_benchmark.py --episodes 160 --seed 42

# Comparacion unificada
uv run python run_all_comparison.py --methods ddpg td3 sac --output-dir outputs

# Agregacion y reporte
uv run python scripts/aggregate_results.py --inputs outputs/*.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Continuous Control Aggregate Report"
```

## Recomendacion practica
1. Comenzar con SAC como baseline robusto.
2. Comparar con TD3 para estudiar estabilidad con policy determinista.
3. Mantener DDPG como baseline pedagogico y de ablation.

## Referencias
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
