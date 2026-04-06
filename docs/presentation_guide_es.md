# RL Algoritmos Hibridos - Guia de Presentacion (60 minutos)

## Alcance
Esta sesion cubre tres algoritmos actor-critic off-policy para espacios de accion continuos:
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)

Enfoque de la sesion:
- Por que DQN no escala bien a acciones continuas.
- Como el actor reemplaza `argmax_a Q(s,a)`.
- Que problemas corrige TD3 respecto a DDPG.
- Como SAC integra exploracion mediante entropia.
- Como mapear teoria directamente al codigo final del repositorio.

---

## 1) Objetivo
Al terminar la presentacion, la audiencia deberia poder:
1. Explicar por que el control continuo requiere actor-critic.
2. Describir el loop de entrenamiento de DDPG, TD3 y SAC.
3. Entender diferencias de estabilidad y exploracion entre los tres metodos.
4. Ejecutar y comparar los algoritmos con los scripts de `rl_continuous_optimization`.

---

## 2) Agenda sugerida (60 min)
- 0-5 min: limitaciones de DQN en continuo.
- 5-22 min: DDPG (actor, critic, replay, target networks).
- 22-37 min: TD3 (twin Q, delayed update, smoothing).
- 37-52 min: SAC (policy estocastica + entropia).
- 52-57 min: comparacion y recomendaciones practicas.
- 57-60 min: Q&A.

---

## 3) Problema de acciones continuas
Con acciones discretas, DQN evalua todas y toma `argmax`.
Con acciones continuas, eso no es viable porque hay infinitas acciones posibles.

Idea clave:
- DQN: `a* = argmax_a Q(s,a)`
- Actor-critic continuo: `a* ~= mu(s)` o `a ~ pi(.|s)`

En otras palabras, una red actor produce la accion, y una red critic evalua su calidad.

---

## 4) DDPG

### Intuicion
DDPG aprende:
- Actor determinista `mu(s)`
- Critic `Q(s,a)`

### Loop esencial
1. Ejecutar accion con ruido de exploracion.
2. Guardar transicion en replay buffer.
3. Muestrear minibatch aleatorio.
4. Actualizar critic con target Bellman.
5. Actualizar actor maximizando Q estimada por el critic.
6. Actualizar target networks con Polyak (`tau`).

### Update del actor en codigo final
```python
# rl_continuous_optimization/benchmarks/ddpg.py
actor_loss = -critic(s, actor(s)).mean()
```

### Mapa de codigo
- Runner individual: `../rl_continuous_optimization/ddpg_benchmark.py`
- Implementacion: `../rl_continuous_optimization/benchmarks/ddpg.py`
- Replay buffer: `../rl_continuous_optimization/benchmarks/replay_buffer.py`
- Runner unificado: `../rl_continuous_optimization/run_all_comparison.py`

### Defaults principales
- `env=Pendulum-v1`
- `episodes=160`
- `gamma=0.99`
- `tau=0.005`
- `batch_size=256`
- `warmup_steps=5000`

---

## 5) TD3

### Problema que corrige
DDPG puede sobreestimar Q-values. TD3 agrega tres mejoras:
1. **Twin critics** y target con `min(Q1, Q2)`.
2. **Delayed policy updates** (actor menos frecuente).
3. **Target policy smoothing** (ruido clippeado en accion target).

### Fragmento representativo
```python
# rl_continuous_optimization/benchmarks/td3.py
current_q1, current_q2 = critic(s, a)
next_action = (actor_target(ns) + noise).clamp(-max_action, max_action)
target_q = r + nd * gamma * torch.min(target_q1, target_q2)

if update_step % policy_delay == 0:
    actor_loss = -critic.q1_only(s, actor(s)).mean()
```

### Mapa de codigo
- Runner individual: `../rl_continuous_optimization/td3_benchmark.py`
- Implementacion: `../rl_continuous_optimization/benchmarks/td3.py`
- Runner unificado: `../rl_continuous_optimization/run_all_comparison.py`

---

## 6) SAC

### Intuicion
SAC optimiza retorno y entropia al mismo tiempo.
Esto mantiene exploracion estructurada mediante una policy estocastica.

### Objetivo clave
- Critic target incluye termino de entropia `-alpha * log pi(a|s)`.
- Actor optimiza equilibrio entre valor Q y entropia.

### Fragmento representativo
```python
# rl_continuous_optimization/benchmarks/sac.py
next_a, next_logp = actor.sample(ns, deterministic=False)
target_q = torch.min(target_q1, target_q2) - alpha * next_logp
target = r + nd * gamma * target_q

pi_action, logp_pi = actor.sample(s, deterministic=False)
actor_loss = (alpha * logp_pi - min_q_pi).mean()
```

### Mapa de codigo
- Runner individual: `../rl_continuous_optimization/sac_benchmark.py`
- Implementacion: `../rl_continuous_optimization/benchmarks/sac.py`
- Runner unificado: `../rl_continuous_optimization/run_all_comparison.py`

### Nota de arquitectura
La implementacion final usa SAC estilo Twin-Q + actor estocastico (sin red V separada), con `alpha` fijo por configuracion.

---

## 7) Comparacion rapida
| Caracteristica | DDPG | TD3 | SAC |
|---|---|---|---|
| Policy | Determinista | Determinista | Estocastica |
| Critics | 1 | 2 (twin) | 2 (twin) |
| Exploracion | Ruido externo | Ruido externo | Entropia incorporada |
| Control sobreestimacion | Bajo | Alto | Alto |
| Robustez tipica | Media-Baja | Media-Alta | Alta |

Recomendacion practica:
1. Empezar con SAC como baseline robusto.
2. Probar TD3 para comparacion de estabilidad sin entropia.
3. Usar DDPG como baseline pedagogico o de referencia.

---

## 8) Comandos de demo
```bash
cd rl_continuous_optimization

# Individuales
uv run python ddpg_benchmark.py --episodes 160
uv run python td3_benchmark.py --episodes 160
uv run python sac_benchmark.py --episodes 160

# Comparacion conjunta
uv run python run_all_comparison.py --methods ddpg td3 sac --output-dir outputs

# Agregacion multi-seed y reporte
uv run python scripts/aggregate_results.py --inputs outputs/*.json --output-json outputs/aggregate_summary.json --output-csv outputs/aggregate_summary.csv
uv run python scripts/generate_aggregate_report.py --input outputs/aggregate_summary.json --output-dir outputs/report --title "RL Continuous Control Aggregate Report"
```

---

## 9) Fuentes recomendadas
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
