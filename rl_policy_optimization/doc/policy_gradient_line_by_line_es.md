# `benchmarks/policy_gradient.py` — Explicación línea por línea (Español)

Este documento complementario explica qué hace cada sección de `benchmarks/policy_gradient.py` y por qué es importante desde el punto de vista matemático.

Archivo principal de código:
- [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)

Guía de presentación (Parte 1):
- [presentation_guide_60min_es.md](presentation_guide_60min_es.md)

---

## 1) Imports y propósito del módulo

- `from __future__ import annotations`
  - Permite evaluación diferida de type annotations; mantiene hints más flexibles.

- Imports estándar (`dataclass`, `gymnasium`, `numpy`, `torch`, etc.)
  - `gymnasium`: API del entorno (`reset`, `step`)
  - `numpy`: agregación de recompensas
  - `torch`: red neuronal, autodiff, optimizer

- `from benchmarks.common import record_policy_video`
  - Utilidad compartida para guardar rollouts de evaluación en video.

Relación con teoría de Parte 1:
- Este archivo implementa la forma práctica de las secciones “Policy gradient theorem”, “Monte Carlo en REINFORCE” y “Algoritmo REINFORCE” en [presentation_guide_60min_es.md](presentation_guide_60min_es.md).

---

## 2) `PolicyGradientConfig` (controles de entrenamiento)

`PolicyGradientConfig` centraliza hiperparámetros:

- `env_name`
  - Tarea Gym sobre la que se entrena (`CartPole-v1` por defecto).

- `gamma`
  - Discount factor de la definición de return en Parte 1:
  - $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$

- `learning_rate`
  - Tamaño de paso del optimizador Adam.

- `episodes`
  - Número de episodios Monte Carlo para aprender.

- `hidden_size`
  - Ancho de la primera capa oculta de la red de policy.

- `normalize_returns`
  - Trick de estabilización explicado en sección Monte Carlo.

- `record_video`, `video_dir`, `video_episodes`
  - Grabación opcional de evaluación post-entrenamiento.

---

## 3) `PolicyNetwork` (estado -> logits de acción)

### Constructor
- Construye un MLP con ReLU y capa final lineal al número de acciones.
- La salida son **logits**, no probabilidades.

### `forward`
- Pasa la entrada por la red y devuelve logits.

Relación con teoría:
- En Parte 1, la policy es $\pi_\theta(a|s)$.
- En código, $\theta$ son todos los parámetros entrenables de esta red.
- Las probabilidades se calculan luego con `softmax(logits)`.

---

## 4) `_discounted_returns` (reward-to-go)

Este helper calcula $G_t$ para cada timestep de un episodio.

Mecánica:
1. Recorre recompensas en reversa.
2. Mantiene acumulador: `running = r + gamma * running`.
3. Guarda cada valor acumulado.
4. Invierte la lista para volver al orden temporal.

¿Por qué en reversa?
- Forma recursiva natural del return:
- $G_t = r_t + \gamma G_{t+1}$

Conexión con Parte 1:
- Es exactamente la variante “reward-to-go” de la sección de variantes del policy gradient.

---

## 5) `run_policy_gradient` — pipeline completo

### 5.1 Setup
- `cfg = config or PolicyGradientConfig()`
  - Usa configuración provista o defaults.

- `env = gym.make(cfg.env_name)`
  - Crea instancia del entorno.

- `obs_size`, `n_actions`
  - Lee dimensiones de observación y acción desde spaces del entorno.

- `net = PolicyNetwork(...)`
  - Instancia el modelo de policy.

- `optimizer = optim.Adam(...)`
  - Optimizador que actualiza parámetros de red.

### 5.2 Loop de episodios (muestreo Monte Carlo)
- Para cada episodio:
  - reset de entorno
  - rollout hasta terminal/truncated
  - recolecta:
    - `log_probs`: $\log \pi_\theta(a_t|s_t)$
    - `rewards`: secuencia de recompensas

Dentro de `while not done`:
- `state_t = torch.tensor(...).unsqueeze(0)`
  - Convierte estado a tensor y agrega dimensión batch.

- `logits = net(state_t)`
  - Forward pass.

- `probs = torch.softmax(logits, dim=1)`
  - Convierte logits en distribución de probabilidad válida.

- `dist = torch.distributions.Categorical(probs)`
  - Representa policy estocástica para acciones discretas.

- `action = dist.sample()`
  - Muestrea acción según policy (clave para estimador insesgado).

- `env.step(int(action.item()))`
  - Ejecuta acción y devuelve siguiente estado/recompensa.

- `done = terminated or truncated`
  - Maneja las dos señales de finalización de Gymnasium.

- `log_probs.append(dist.log_prob(action))`
  - Guarda término exacto que requiere policy-gradient theorem.

- `rewards.append(float(reward))`
  - Guarda recompensa escalar para calcular returns.

### 5.3 Construcción de loss REINFORCE
Al finalizar episodio:

- `returns = _discounted_returns(rewards, cfg.gamma)`
  - Calcula reward-to-go $G_t$.

- `returns_t = torch.tensor(returns, dtype=torch.float32)`
  - Convierte a tensor para operaciones vectorizadas.

- Normalización opcional:
  - `(returns_t - mean) / (std + 1e-8)`
  - Reduce volatilidad de escala del gradiente.

- `policy_loss = -torch.stack(log_probs) * returns_t`
  - Término REINFORCE por paso:
  - $-\log\pi_\theta(a_t|s_t) G_t$

- `loss = policy_loss.sum()`
  - Suma sobre todos los timesteps del episodio.

¿Por qué signo menos?
- El optimizer hace gradient **descent** sobre `loss`.
- Nosotros queremos gradient **ascent** sobre return esperado.
- La negación vuelve equivalentes ambos objetivos.

### 5.4 Backpropagation y update
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

Esto calcula y aplica update de parámetros de la policy.

### 5.5 Tracking de recompensas
- `ep_reward = np.sum(rewards)`
- `episode_rewards.append(ep_reward)`

Se usa luego para métricas de benchmark (`max_avg_reward_100`, `final_avg_reward_100`).

### 5.6 Grabación opcional de video
Si `cfg.record_video`:
- `net.eval()` pone modelo en modo evaluación.
- Define `_policy` determinística con `argmax(logits)`.
- Llama `record_policy_video(...)`.

Distinción importante:
- Entrenamiento usa acciones estocásticas muestreadas.
- Video de evaluación usa policy greedy determinística para demostración reproducible.

### 5.7 Cierre y retorno
- `env.close()` libera recursos del entorno.
- `return episode_rewards` entrega historial de recompensas al runner de benchmark.

---

## 6) Mapeo directo a explicaciones de Parte 1

Usa estos enlaces durante la presentación:

- Objetivo y gradient ascent:
  - [presentation_guide_60min_es.md](presentation_guide_60min_es.md)
  - Secciones: “Policy optimization desde primeros principios”, “Policy gradient theorem”.

- Monte Carlo y reward-to-go:
  - [presentation_guide_60min_es.md](presentation_guide_60min_es.md)
  - Secciones: “Monte Carlo en REINFORCE”, “Variantes del policy gradient”.

- Expresión de loss y optimizer step:
  - [presentation_guide_60min_es.md](presentation_guide_60min_es.md)
  - Secciones: “Algoritmo REINFORCE (paso a paso)”.

- Código explicado:
  - [benchmarks/policy_gradient.py](../benchmarks/policy_gradient.py)
