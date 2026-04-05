# PPO (Proximal Policy Optimization)

## Idea central
PPO restringe policy updates usando un surrogate objective con clipping para evitar updates grandes destructivos, manteniendo optimización simple.

Objetivo con clipping:

$$
L^{CLIP}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)A_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]
$$

donde

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

Intuición:
- Si la policy nueva cambia demasiado la probabilidad de una acción, el clipping recorta el incentivo para seguir empujando ese cambio.
- Esto crea una especie de trust region "suave" sin resolver la optimización restringida completa de TRPO.

## Cómo lo implementa este repo
- Módulo: `benchmarks/ppo.py`
- Runner: `ppo_benchmark.py`
- Usa rollout collection + GAE + múltiples epochs por minibatch.

## Flujo de entrenamiento
1. Recolectar transiciones con policy vieja.
2. Calcular advantages con GAE.
3. Normalizar advantages.
4. Ejecutar varias epochs de update por minibatches con clip loss.
5. Optimizar de forma conjunta objetivos de policy y value (+ entropy bonus).

PPO suele ser sensible a la interacción entre rollout size, minibatch size y número de epochs por update.

## Fortalezas
- Robusto y muy usado en práctica.
- Mejor estabilidad que policy gradients simples.
- Complejidad de implementación razonable.

## Limitaciones
- Sigue siendo on-policy; sample efficiency puede quedar por detrás de métodos off-policy.
- Sensible a rollout size, cantidad de epochs y clip range.

## Palancas de ajuste comunes
- **Clip range ($\epsilon$)**: más bajo = updates más conservadores; más alto = updates más rápidos pero más riesgosos.
- **Epochs por rollout**: demasiadas pueden sobreajustar datos ya viejos.
- **GAE $\lambda$**: controla el trade-off bias-varianza en advantages.
- **Coeficiente de value loss**: muy alto puede opacar la mejora de policy.

## Notas prácticas
- Fallo común: demasiadas epochs de update sobreajustan rollouts viejos.
- Mantener preprocessing de observation/reward consistente para comparaciones justas.
- Si la KL divergence sube bruscamente, bajar learning rate o clip range.

---

## Anclas de código en el repositorio (mapeo práctico)
- Implementación principal: [../../benchmarks/ppo.py](../../benchmarks/ppo.py)
- Runner standalone: [../../ppo_benchmark.py](../../ppo_benchmark.py)
- Orquestador de comparación: [../../run_all_comparison.py](../../run_all_comparison.py)

Símbolos clave para presentar en vivo:
- `PPOConfig`
- `_compute_gae(...)`
- `run_ppo(...)`

---

## Fuentes externas sugeridas
- https://huggingface.co/blog/deep-rl-ppo
- https://docs.pytorch.org/rl/0.7/tutorials/multiagent_ppo.html
