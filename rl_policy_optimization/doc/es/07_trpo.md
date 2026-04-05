# TRPO (Trust Region Policy Optimization)

## Idea central
TRPO impone un trust-region constraint sobre policy updates limitando la KL divergence entre policy vieja y nueva:

$$
\max_\theta\; \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}A(s,a)\right]
\quad\text{s.t.}\quad
\mathbb{E}[D_{KL}(\pi_{old}\|\pi_\theta)] \leq \delta
$$

Comparado con PPO, TRPO resuelve un problema de optimización más restringido (normalmente con conjugate gradient + line search).

Intuición:
- TRPO busca la mejor mejora esperada manteniendo la policy nueva cerca de la policy vieja en el espacio de distribuciones.
- La cota KL $\delta$ actúa como presupuesto de seguridad por update.

## Cómo lo implementa este repo
- Módulo: `benchmarks/trpo.py`
- Runner: `trpo_benchmark.py`
- Usa implementación TRPO de `sb3-contrib` para confiabilidad y mantenibilidad.

## Fortalezas
- Perspectiva sólida de mejora monotónica.
- Updates estables con control explícito de trust-region.

## Perspectiva de optimización
En práctica, TRPO aproxima información de segundo orden (vía Fisher-vector products) y usa line search para respetar la restricción KL. Por eso puede ser más estable por paso, pero más pesado computacionalmente que métodos de primer orden como PPO.

## Limitaciones
- Internamente más complejo que PPO.
- Mayor carga de implementación cuando se desarrolla desde cero.

## Notas prácticas
- Este repositorio delega internals de optimización en `sb3-contrib`.
- Si falta la dependencia, el script muestra un mensaje claro de instalación.
- Para workflows de producción, PPO suele preferirse por simplicidad y velocidad.
- Si TRPO resulta demasiado lento para iteración rápida, usar PPO como baseline por defecto y reservar TRPO para escenarios donde se prioriza control conservador del update.

---

## Anclas de código en el repositorio (mapeo práctico)
- Wrapper principal de integración: [../../benchmarks/trpo.py](../../benchmarks/trpo.py)
- Runner standalone: [../../trpo_benchmark.py](../../trpo_benchmark.py)
- Orquestador de comparación: [../../run_all_comparison.py](../../run_all_comparison.py)

Símbolos clave para presentar en vivo:
- `TRPOConfig`
- `EpisodeRewardCallback`
- `run_trpo(...)`

---

## Fuentes externas sugeridas
- https://dilithjay.com/blog/trpo
- https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9
- https://towardsdatascience.com/trust-region-policy-optimization-trpo-explained-4b56bd206fc2/
