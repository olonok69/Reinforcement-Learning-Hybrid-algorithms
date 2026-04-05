# RL Policy Optimization — Guía de Presentación Parte 2 (60 minutos)

## Alcance de la Parte 2
Esta sesión cubre los algoritmos restantes después de REINFORCE:
- A2C
- A3C
- PPO
- TRPO

Enfoque:
- por qué estos métodos mejoran estabilidad/eficiencia frente a REINFORCE
- cómo mapear cada método al código del repositorio
- cómo comparar métodos con un flujo de benchmark reproducible

---

## 1) Objetivo de la sesión
Al terminar esta Parte 2, la audiencia debería entender:
- el rol de critic baselines para reducir varianza
- por qué la recolección asíncrona puede ayudar (A3C)
- por qué clipping/trust-region estabiliza entrenamiento (PPO/TRPO)
- cómo ejecutar y comparar todos los métodos en este repositorio

---

## 2) Agenda sugerida (60 minutos)

- **0–6 min**: recap rápido de Parte 1 (limitaciones de REINFORCE)
- **6–20 min**: A2C (actor + critic)
- **20–30 min**: A3C (workers asíncronos)
- **30–43 min**: PPO (surrogate con clipping + GAE)
- **43–50 min**: TRPO (concepto trust-region + integración de librería)
- **50–57 min**: flujo de ejecución/comparación de benchmark
- **57–60 min**: resumen de recomendaciones + Q&A

---

## 3) A2C (Advantage Actor-Critic)

### Intuición principal
REINFORCE usa returns completos directos y puede ser ruidoso.
A2C introduce baseline de valor `V(s)` y usa advantage:

```
Aₜ = Rₜ - V(sₜ)
```

Esto normalmente reduce varianza del gradiente.

**Analogía (de HuggingFace Deep RL):** Imagina que juegas un videojuego. Tú eres el **Actor** (juegas y eliges acciones). Tu amigo sentado al lado es el **Critic** (observa y te dice "buen movimiento" o "eso fue terrible"). Al principio no sabes jugar, así que pruebas acciones al azar. El Critic observa y da feedback. Con ese feedback, actualizas tu estrategia. Mientras tanto, el Critic también mejora su capacidad de juzgar.

### Error TD como estimador de ventaja
En la práctica, calcular la ventaja exacta `A = Q(s,a) - V(s)` requiere dos redes. Un enfoque más simple usa el **error TD** como aproximación:

```
δₜ = rₜ + γ · V(sₜ₊₁) - V(sₜ)
```

Esto dice: "¿la recompensa real + valor del siguiente estado fue mejor o peor que lo que predije para este estado?" Si δ > 0, la acción fue mejor que lo esperado. Es un estimador de 1 paso de la ventaja — sesgado pero con varianza mucho menor que returns Monte Carlo completos.

### Mapa de código
- Algoritmo: [benchmarks/a2c.py](../benchmarks/a2c.py)
- Runner standalone: [a2c_benchmark.py](../a2c_benchmark.py)

### Qué señalar en el código
- shared backbone + policy/value heads
- policy loss + value loss + entropy term
- cálculo de returns y advantage por episodio

### Concepto clave → ancla exacta en código
- Definición del modelo: `ActorCritic` en [../benchmarks/a2c.py](../benchmarks/a2c.py)
- Entrypoint principal: `run_a2c(...)` en [../benchmarks/a2c.py](../benchmarks/a2c.py)
- Cálculo de advantage: `advantages_t = returns_t - values_t.detach()` en [../benchmarks/a2c.py](../benchmarks/a2c.py)
- Losses de policy/value/entropía: `policy_loss`, `value_loss`, `entropy_t` en [../benchmarks/a2c.py](../benchmarks/a2c.py)

### Cómo funciona A2C paso a paso

```
1. Recolectar un episodio completo con la policy actual
2. Calcular returns descontados Monte Carlo Gₜ para cada paso
3. Obtener V(sₜ) de la cabeza critic para cada estado
4. Calcular advantage: Aₜ = Gₜ - V(sₜ)
5. Calcular loss combinado:
     L = -Σ log π(a|s) · A        (policy — ponderado por advantage)
       + value_coef · (V(s) - G)²  (critic — aprender a predecir returns)
       - entropy_coef · H(π)       (entropía — prevenir colapso)
6. Backpropagate y actualizar ambas cabezas conjuntamente
7. Repetir por muchos episodios
```

### Correspondencia teoría-código

| Concepto teórico | Código en `benchmarks/a2c.py` |
|-----------------|------------------------------|
| `π(a\|s)` — actor | `ActorCritic.policy_head` → `Categorical(logits)` |
| `V(s)` — critic | `ActorCritic.value_head` → escalar |
| `A = G - V(s)` — advantage | `advantage = returns - values.detach()` |
| Policy loss | `-(log_probs * advantage).sum()` |
| Value loss | `F.mse_loss(values, returns)` |
| Entropy bonus | `dist.entropy().mean()` |
| Loss combinado | `policy_loss + 0.5*value_loss - 0.01*entropy` |

### Qué arregla A2C vs REINFORCE
- **Menor varianza** — el baseline de advantage elimina ruido de returns crudos
- **Mejor asignación de crédito** — solo acciones sobre el promedio se refuerzan
- **Protección de exploración** — bonus de entropía previene colapso prematuro
- **Sigue siendo on-policy** — cada trayectoria se usa una vez y se descarta

### ¿Para qué sirve la entropía en A2C y A3C? (en lenguaje simple)
- Es un **bono de “curiosidad”**: recompensa un poco a la policy por no volverse demasiado rígida muy pronto.
- Sin entropía, el agente puede obsesionarse temprano con una sola acción y dejar de explorar alternativas.
- Con entropía, mantiene más diversidad de acciones al inicio, descubre mejores estrategias y evita atascarse.
- En la loss aparece restando `entropy_coef * entropy`: eso empuja a mantener distribución de acciones más “abierta”.
- Regla práctica: al principio suele ayudar más (exploración); más tarde puedes bajar `entropy_coef` para volver la policy más decidida.

---

## 4) A3C (Asynchronous Advantage Actor-Critic)

### Intuición principal
A3C mantiene estructura actor-critic pero recolecta experiencia con múltiples workers en paralelo.

**Por qué importa (de Arthur Juliani / paper DeepMind 2016):** En DQN, un solo agente interactúa con un solo entorno — la experiencia está altamente correlacionada (estados consecutivos son similares). DQN resuelve esto con un replay buffer. A3C toma un enfoque completamente diferente: en vez de guardar y repetir experiencia vieja, **ejecuta múltiples agentes en paralelo**, cada uno en su propia copia del entorno. Como cada worker explora desde estados diferentes simultáneamente, el batch de transiciones recolectado está naturalmente decorrelacionado — **no necesita replay buffer**.

**El resultado histórico:** En el paper original de 2016, A3C resolvió los mismos juegos Atari que DQN usando solo **16 cores de CPU** en vez de una GPU potente — logrando mejor rendimiento en **1 día** vs los 8 días de DQN. El speedup es casi lineal: más workers → datos más diversos → convergencia más rápida.

Beneficios:
- decorrelación de datos (reemplaza el replay buffer)
- mejora de tiempo de pared en CPU
- exploración diversa (cada worker ve estados diferentes)

Trade-off:
- más complejidad de sistema (procesos, colas, sincronización)
- los workers pueden tener parámetros algo desactualizados (policy lag)

### Mapa de código
- Algoritmo: [benchmarks/a3c.py](../benchmarks/a3c.py)
- Runner standalone: [a3c_benchmark.py](../a3c_benchmark.py)

### Qué señalar en el código
- worker loop con episodios y mini-rollouts
- learner consumiendo batches desde cola
- patrón de refresco de parámetros compartidos

### Concepto clave → ancla exacta en código
- Lógica de rollout de workers: `_worker_loop(...)` en [../benchmarks/a3c.py](../benchmarks/a3c.py)
- Snapshot de parámetros compartidos: `_snapshot_state_dict(...)` en [../benchmarks/a3c.py](../benchmarks/a3c.py)
- Loop del learner y consumo de cola: `run_a3c(...)` en [../benchmarks/a3c.py](../benchmarks/a3c.py)
- Control asíncrono: `data_queue`, `error_queue`, `stop_event` en [../benchmarks/a3c.py](../benchmarks/a3c.py)

### Arquitectura en detalle

```
┌─────────────┐     ┌─────────────┐
│  Worker 1   │     │  Worker 2   │    ... N workers
│  (su env)   │     │  (su env)   │
│  recolecta  │     │  recolecta  │
│  rollouts   │     │  rollouts   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └──────┬────────────┘
              ▼
       ┌──────────────┐
       │ Cola Compartida│
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │   Learner     │
       │  aplica grads │
       │  actualiza    │
       └──────────────┘
```

Cada worker: ejecuta su propia copia del env → recolecta `rollout_steps` transiciones → calcula advantages → envía batch a la cola.

Learner: desencola batches → aplica loss combinado (igual que A2C) → actualiza parámetros compartidos.

Los workers refrescan periódicamente sus pesos locales desde el modelo compartido.

### Config por defecto
`workers=4, rollout_steps=5, γ=0.99, lr=1e-3, value_coef=0.5, entropy_coef=0.01`

### A2C vs A3C
- **A2C** — updates síncronos, un solo proceso, más fácil de debuggear y reproducir
- **A3C** — workers asíncronos, multiprocessing, mejor velocidad en CPUs multi-core
- **Trade-off**: multiprocessing agrega complejidad de ingeniería (colas, sincronización, manejo de errores)

Nota: en **A3C** la entropía cumple exactamente el mismo objetivo que en A2C (evitar colapso prematuro y sostener exploración), solo que ahora se aplica sobre datos que llegan de múltiples workers.

### Config por defecto
`workers=4, rollout_steps=5, γ=0.99, lr=1e-3, value_coef=0.5, entropy_coef=0.01`

### A2C vs A3C
- **A2C** — updates síncronos, un solo proceso, más fácil de debuggear y reproducir
- **A3C** — workers asíncronos, multiprocessing, mejor velocidad en CPUs multi-core
- **Trade-off**: multiprocessing agrega complejidad de ingeniería (colas, sincronización, manejo de errores)

---

## 5) PPO (Proximal Policy Optimization)

### Intuición principal
PPO evita saltos grandes de policy con objective clippeado:

```
L_clip(θ) = E[ min( rₜ(θ)·Aₜ , clip(rₜ(θ), 1-ε, 1+ε)·Aₜ ) ]

donde rₜ(θ) = π_new(aₜ|sₜ) / π_old(aₜ|sₜ)
```

Suele ser default práctico por balance entre estabilidad y simplicidad.

**La analogía del "precipicio" (de HuggingFace Deep RL):** Imagina estar en la ladera de una montaña. El gradiente te dice "da un paso a la derecha". Un paso normal está bien — te acercas a la cima. Pero un paso apenas más grande te lanza por un precipicio a un valle completamente diferente, y toma mucho tiempo volver a subir. En supervised learning, otros datos te corrigen. En RL, **los datos dependen de tu policy actual** — si das un mal paso, tus datos futuros vienen de una policy mala, creando una espiral descendente. PPO previene esto limitando el tamaño de cada paso.

### Los 6 casos del clipping explicados

La fórmula `min(unclipped, clipped)` crea 6 comportamientos según el ratio `r` y la ventaja `A`:

```
Caso 1: r en [0.8, 1.2], A > 0  →  gradiente empuja acción ARRIBA (update normal)
Caso 2: r en [0.8, 1.2], A < 0  →  gradiente empuja acción ABAJO (update normal)
Caso 3: r < 0.8,         A > 0  →  gradiente empuja ARRIBA (quiere recuperar)
Caso 4: r < 0.8,         A < 0  →  gradiente = 0 (ya está suficientemente desalentada)
Caso 5: r > 1.2,         A > 0  →  gradiente = 0 (ya está suficientemente alentada)
Caso 6: r > 1.2,         A < 0  →  gradiente empuja ABAJO (quiere corregir)
```

**Insight clave:** En los casos 4 y 5, el gradiente es CERO — la policy ya se movió suficiente en esa dirección, así que el clip detiene más movimiento. Este es el mecanismo que previene updates catastróficos.

### Mapa de código
- Algoritmo: [benchmarks/ppo.py](../benchmarks/ppo.py)
- Runner standalone: [ppo_benchmark.py](../ppo_benchmark.py)

### Qué señalar en el código
- recolección de rollouts (1024 pasos)
- cálculo de GAE (`_compute_gae()`)
- updates por minibatch y múltiples epochs con clipping

### Concepto clave → ancla exacta en código
- Función GAE: `_compute_gae(...)` en [../benchmarks/ppo.py](../benchmarks/ppo.py)
- Entrenador principal: `run_ppo(...)` en [../benchmarks/ppo.py](../benchmarks/ppo.py)
- Surrogate clippeado: `ratio`, `surr1`, `surr2`, `policy_loss` en [../benchmarks/ppo.py](../benchmarks/ppo.py)
- Update multi-epoch por minibatch: loop `for start in range(...)` en [../benchmarks/ppo.py](../benchmarks/ppo.py)

### El mecanismo de clipping explicado

El ratio de probabilidad mide cuánto cambió la policy:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- `r = 1.0` → policy sin cambios
- `r = 1.5` → acción 50% más probable bajo nueva policy
- `r = 0.5` → acción 50% menos probable

Clipear a `[1-ε, 1+ε]` (default `[0.8, 1.2]`) limita cuánto puede empujar un solo update:

```python
# benchmarks/ppo.py — el clipping central
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantage
policy_loss = -torch.min(surr1, surr2).mean()
```

### GAE (Generalized Advantage Estimation) — explicado desde cero

GAE resuelve un problema fundamental: ¿cómo estimamos "¿fue buena esta acción?"? Hay dos respuestas extremas, y GAE las mezcla.

**El problema: sesgo vs varianza**

Para calcular advantage `A(s,a)`, necesitamos saber cuánto return produjo una acción vs cuánto esperábamos. Pero podemos estimar esto a diferentes horizontes:

```
TD de 1 paso: δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)
              "Mira 1 paso adelante, luego confía en V para el resto"
              Baja varianza (un reward real) pero alto sesgo (V puede estar mal)

MC completo:  Aₜ = Gₜ - V(sₜ)  =  (rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...) - V(sₜ)
              "Espera al final del episodio, usa todos los rewards reales"
              Sin sesgo (todos rewards reales) pero alta varianza (trayectorias ruidosas)
```

**GAE: lo mejor de ambos mundos**

GAE calcula un promedio ponderado de advantages n-step usando parámetro `λ`:

```
GAE(λ) = δₜ + (γλ)·δₜ₊₁ + (γλ)²·δₜ₊₂ + ...

donde cada δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)   (error TD de 1 paso)
```

Piénsalo así: empieza con el estimado de 1 paso `δₜ`, luego mezcla correcciones de pasos futuros, pero da peso exponencialmente decreciente `(γλ)ᵏ` a cada paso adicional.

**Qué controla λ:**

```
λ = 0:     GAE = solo δₜ           → TD puro de 1 paso (menor varianza, mayor sesgo)
λ = 0.5:   GAE = mezcla de ~2-3 pasos → compromiso moderado
λ = 0.95:  GAE ≈ mayormente Monte Carlo → bajo sesgo, varianza moderada (el punto óptimo)
λ = 1:     GAE = returns MC completos → sin sesgo, mayor varianza
```

**Ejemplo numérico** — episodio con 3 pasos, `γ=0.99`, `λ=0.95`:

```
Rewards: [1, 1, 1],  V(s₀)=2.5, V(s₁)=1.8, V(s₂)=0.9, V(s₃)=0

δ₀ = 1 + 0.99·1.8 - 2.5 = 0.282     (paso 0 ligeramente sobre lo esperado)
δ₁ = 1 + 0.99·0.9 - 1.8 = 0.091     (paso 1 más o menos esperado)
δ₂ = 1 + 0.99·0   - 0.9 = 0.100     (paso 2 ligeramente sobre lo esperado)

GAE₂ = δ₂                          = 0.100
GAE₁ = δ₁ + 0.99·0.95·GAE₂        = 0.091 + 0.094 = 0.185
GAE₀ = δ₀ + 0.99·0.95·GAE₁        = 0.282 + 0.174 = 0.456
```

El paso 0 recibe la mayor ventaja porque captura el rendimiento acumulado por encima del promedio de todos los pasos futuros.

**El código — caminando hacia atrás por el rollout:**

```python
# benchmarks/ppo.py Y benchmarks/trpo_native.py — _compute_gae()
def _compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):       # caminar HACIA ATRÁS
        not_done = 1.0 - float(dones[t])
        next_v = next_value if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + gamma * next_v * not_done - values[t]  # error TD
        gae = delta + gamma * gae_lambda * not_done * gae           # acumular
        advantages[t] = gae
    returns = advantages + values                  # targets para red de valor
    return advantages, returns
```

El loop camina hacia atrás porque cada `GAE[t]` depende de `GAE[t+1]` — exactamente como calcular returns descontados, pero con el decay extra de `λ` y errores TD en vez de rewards crudos.

**Dónde aparece GAE en este repo:**
- PPO: `_compute_gae()` en `benchmarks/ppo.py` (λ=0.95)
- TRPO nativo: `_compute_gae()` en `benchmarks/trpo_native.py` (λ=0.97)
- A2C: NO usa GAE — usa el más simple `A = G - V(s)` con returns MC completos

### Flujo de entrenamiento
1. Recolectar rollout (1024 pasos) con policy actual
2. Calcular advantages con GAE
3. Normalizar advantages
4. Ejecutar 4 epochs por minibatch (shuffle + split en batches de 64) con loss clipeado
5. Descartar rollout, repetir

### Config por defecto
`rollout_steps=1024, update_epochs=4, minibatch_size=64, clip_eps=0.2, gae_lambda=0.95, lr=3e-4, value_coef=0.5, entropy_coef=0.01`

### Por qué PPO es el default práctico
- **Estable** — clipping previene updates catastróficos
- **Simple** — no requiere optimización restringida (a diferencia de TRPO)
- **Reutiliza datos** — múltiples epochs por rollout (más sample-efficient que A2C)
- **RLHF** — usado para finetunear ChatGPT, LLaMA, y otros LLMs

---

## 6) TRPO (Trust Region Policy Optimization)

### Intuición principal
TRPO restringe updates para mantener la policy en una trust region (típicamente con constraint KL).

**La analogía de la montaña (de Dilith Jayakody):** Imagina estar en la ladera de una montaña con forma extraña. El gradiente te dice "da un paso a la derecha". Un paso normal te acerca al valle (bien). Pero un paso apenas más grande te lanza a un pozo completamente diferente — y recuperarse es muy difícil. En supervised learning, otros datos etiquetados te corrigen. Pero en RL, **los datos son no-estacionarios**: dependen de tu policy actual. Si un update malo lleva a acciones malas, todos los datos futuros de entrenamiento vienen de esas acciones malas, creando un ciclo vicioso. TRPO previene esto definiendo una "trust region" — el espacio alrededor de tu policy actual donde confías que los updates son seguros.

**Por qué RL es más difícil que supervised learning aquí:** En supervised learning, siempre tienes las etiquetas correctas. Aunque un paso del gradiente sea malo, las demás etiquetas te corrigen. En RL, si tu policy da un mal paso, genera trayectorias malas, que producen gradientes malos, que empeoran la policy. La distribución de datos cambia con cada update — este es el problema de **no-estacionaridad** que hace peligrosos los updates grandes.

Pros:
- updates conservadores y principiados
- garantía teórica de mejora monótona

Contras:
- maquinaria de optimización más pesada (gradiente conjugado + backtracking line search)

### Mapa de código
- Router de backends (sb3 + native): [benchmarks/trpo.py](../benchmarks/trpo.py)
- Entrenador TRPO nativo (CartPole discreto): [benchmarks/trpo_native.py](../benchmarks/trpo_native.py)
- Utilidades matemáticas TRPO nativo (CG + line search): [benchmarks/trpo_core.py](../benchmarks/trpo_core.py)
- Runner standalone / bandera de backend: [trpo_benchmark.py](../trpo_benchmark.py)
- Orquestador unificado / bandera de backend: [run_all_comparison.py](../run_all_comparison.py)

### Qué señalar en el código
- Dos rutas de implementación bajo una sola interfaz: `backend="sb3"` vs `backend="native"`
- La ruta nativa implementa explícitamente la mecánica trust-region (gradiente conjugado + backtracking line search)
- Mismo contrato de salida de benchmark para ambos backends (`list[float]` rewards)
- Mismo flujo CLI/orquestador, solo cambia la bandera de backend

### Concepto clave → ancla exacta en código
- Entrypoint de despacho por backend: `run_trpo(...)` con `cfg.backend` en [../benchmarks/trpo.py](../benchmarks/trpo.py)
- Entrypoint del entrenador nativo: `run_trpo_native(...)` en [../benchmarks/trpo_native.py](../benchmarks/trpo_native.py)
- Resolución trust-region nativa: `conjugate_gradients(...)` y `backtracking_line_search(...)` en [../benchmarks/trpo_core.py](../benchmarks/trpo_core.py)
- Internals del paso nativo: `fisher_vector_product`, `step_dir`, `full_step` en [../benchmarks/trpo_native.py](../benchmarks/trpo_native.py)
- Switch CLI: `--backend sb3|native` en [../trpo_benchmark.py](../trpo_benchmark.py)
- Switch unificado: `--trpo-backend sb3|native` en [../run_all_comparison.py](../run_all_comparison.py)

### La restricción de trust region

```
maximizar  E[ r(θ) · A ]
sujeto a   KL( π_old || π_new ) ≤ δ
```

KL divergence mide qué tan diferentes son dos distribuciones de probabilidad. La restricción dice: "mejora la policy, pero no la cambies demasiado respecto a la actual."

TRPO resuelve esto exactamente usando gradiente conjugado + line search. Teóricamente riguroso pero computacionalmente costoso por update.

### TRPO vs PPO
- **TRPO** — resuelve la optimización restringida exactamente. Mejora monótona garantizada. Costoso.
- **PPO** — aproxima la misma idea con clipping simple. Mucho más barato, casi igual de estable, más fácil de implementar.
- PPO fue diseñado como alternativa más simple a TRPO. En la práctica, PPO es preferido para la mayoría de tareas.

### Introducción breve a GRPO
GRPO (Group Relative Policy Optimization) es una variante reciente de policy optimization, popular en flujos de RL para LLMs.

Idea central en lenguaje simple:
- En vez de depender fuertemente de un critic de valor, GRPO compara respuestas dentro de un **grupo** de muestras para el mismo prompt/contexto.
- A cada respuesta se le asigna una señal **relativa** (mejor o peor que las otras del grupo), no solo una señal absoluta.
- Luego aplica actualización tipo PPO (ratio + clipping), pero con esa ventaja/recompensa relativa de grupo.

Intuición: no pregunta solo “¿esta acción fue buena?”, sino “¿fue mejor que sus alternativas cercanas en el mismo contexto?”.

### GRPO vs PPO vs TRPO

| Aspecto | GRPO | PPO | TRPO |
|---|---|---|---|
| Señal principal | Ranking/ventaja relativa en grupo | Advantage (usualmente con critic + GAE) | Advantage con restricción KL explícita |
| Control de tamaño de update | Clipping estilo PPO | Clipping | Constraint KL (trust region) |
| Complejidad práctica | Media (sampling/ranking por grupo) | Baja-media | Alta |
| Uso típico | Ajuste de políticas con preferencias comparativas (muy común en LLM RL) | Baseline general en RL práctico | Casos donde se prioriza garantía teórica/conservadora |

Resumen rápido:
- **TRPO**: más riguroso, más caro.
- **PPO**: balance práctico más usado.
- **GRPO**: mantiene estabilidad tipo PPO, pero usa señal relativa de grupo para aprender de comparaciones.

### Cuándo usar TRPO
- Sistemas safety-critical o robótica donde updates conservadores son esenciales
- Cuando se necesita garantía teórica de mejora monótona
- Cuando el presupuesto de cómputo permite el costo extra por update

### Código: benchmarks/trpo.py
Este repo ahora soporta **implementaciones paralelas de TRPO** detrás de una única interfaz de benchmark:
- **Backend SB3** (`sb3-contrib`) para comportamiento base estable de producción.
- **Backend nativo** (`benchmarks/trpo_native.py`) para comparación educativa/experimental y visibilidad completa de internals trust-region.

Para presentación, esto permite mostrar ambos enfoques:
- integración tipo wrapper industrial (`sb3`), y
- internals algorítmicos (`native`) con gradiente conjugado + line search dentro del código del proyecto.

### Comandos demo para comparar backends (solo TRPO)
```bash
uv run python trpo_benchmark.py --backend sb3 --timesteps 20000
uv run python trpo_benchmark.py --backend native --timesteps 20000
```

### Comando unificado (mismo pipeline, solo cambia backend)
```bash
uv run python run_all_comparison.py --methods trpo --trpo-backend sb3 --trpo-timesteps 20000
uv run python run_all_comparison.py --methods trpo --trpo-backend native --trpo-timesteps 20000
```

---

## 7) GRPO (Group Relative Policy Optimization)

### Intuición principal
GRPO es el algoritmo detrás de DeepSeek-R1 y la frontera actual del RL para razonamiento en LLMs. Se basa en el objetivo clipeado de PPO pero **elimina la red critic por completo**, reemplazándola con rewards normalizados por grupo como estimación de ventaja.

**Por qué funciona para LLMs (de Cameron Wolfe):** El critic de PPO existe para reducir varianza en la estimación de advantage. Pero los LLMs son modelos extensamente pre-entrenados que se están finetuneando — el problema de alta varianza es mucho menos severo que en RL desde cero. Además, los LLMs se entrenan mayormente con **recompensas de resultado** (correcto/incorrecto), lo que hace innecesaria la estimación de valor por token. Así que el critic se puede eliminar.

### Cómo funciona GRPO paso a paso

```
1. Para cada prompt, generar G completaciones (un "grupo") de la policy actual
   Ej: G=8 respuestas diferentes a un problema de matemáticas

2. Puntuar cada completación con reward rᵢ
   (ej: 1 si es correcta, 0 si es incorrecta — rewards verificables, sin reward model)

3. Calcular ventaja relativa al grupo:
   Aᵢ = (rᵢ - mean(r)) / std(r)
   La media del grupo ES el baseline. No se necesita red critic V(s).

4. Actualizar policy con clipping estilo PPO + penalización KL contra
   una policy de referencia para prevenir drift:
   J_GRPO = E[ min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A) ] - β·KL(π_θ || π_ref)
```

### PPO vs GRPO

| Aspecto | PPO | GRPO |
|---------|-----|------|
| Ventaja | GAE sobre V(s) por token | Reward normalizado por grupo |
| Critic | Requerido (red grande) | Ninguno |
| Memoria | Policy + critic + reward model (~3x) | Policy + referencia (~2x) |
| Fuente de reward | Reward model (RLHF) | Rewards verificables (RLVR) |
| Cómputo | Mayor | ~50% menos |
| Usado en | ChatGPT, Claude (RLHF) | DeepSeek-R1, Qwen-3, OLMo-3 (razonamiento) |

### RLHF vs RLVR

- **RLHF** (era ChatGPT): reward model aprendido de preferencias humanas. Usa PPO. Riesgo: reward hacking.
- **RLVR** (era DeepSeek-R1): rewards verificables (correcto/incorrecto). Sin reward model. Usa GRPO. Más difícil de manipular.

### Dónde destaca GRPO
- **Razonamiento matemático** — DeepSeekMath: 46.8% → 51.7% en benchmark MATH
- **Generación de código** — verificable con test cases
- **Modelos de razonamiento** — DeepSeek-R1, Qwen-3, OLMo-3
- **Accesibilidad** — HuggingFace TRL tiene `GRPOTrainer` integrado

### La evolución continúa

```
REINFORCE → A2C → PPO → GRPO
(add critic)  (add clip)  (drop critic, group advantage)
```

### Referencias
- Cameron R. Wolfe — GRPO: [cameronrwolfe.substack.com/p/grpo](https://cameronrwolfe.substack.com/p/grpo)
- DeepSeekMath paper: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1 paper: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

---

## 8) Narrativa de comparación lado a lado

| Método | Idea principal | Fortaleza | Trade-off común |
|---|---|---|---|
| A2C | baseline critic + advantage | Más estable que REINFORCE | Sigue siendo on-policy y menos sample-efficient |
| A3C | actor-critic asíncrono multi-worker | Mejor comportamiento en tiempo de pared (CPU) | Complejidad de multiprocessing |
| PPO | clipping + updates por minibatch repetidos | Default práctico fuerte | Sensible a hiperparámetros |
| TRPO | updates restringidos por trust-region | Mejora conservadora estable | Mayor costo computacional |

---

## 9) Comandos de demo (Parte 2)

### Ejecutar cada método
```bash
uv run python a2c_benchmark.py
uv run python a3c_benchmark.py
uv run python ppo_benchmark.py
uv run python trpo_benchmark.py
```

### Ejecutar todos los métodos con un comando
```bash
uv run python run_all_comparison.py --methods policy_gradient a2c a3c ppo trpo
```

### Ejecutar subset con videos
```bash
uv run python run_all_comparison.py --methods a2c ppo trpo --record-video --video-dir videos --video-episodes 2
```

### Ejemplo de alineación de budgets
```bash
uv run python run_all_comparison.py --methods a2c a3c ppo trpo --a2c-episodes 600 --a3c-episodes 600 --ppo-episodes 700 --trpo-timesteps 200000
```

---

## 10) Reportes e interpretación

Métricas principales del comparison runner:
- `max_avg_reward_100`
- `final_avg_reward_100`
- `elapsed_sec`
- `episodes`

Interpretación recomendada:
- comparar capacidad pico (`max_avg_reward_100`)
- comparar estabilidad final (`final_avg_reward_100`)
- comparar costo práctico (`elapsed_sec`)
- repetir en múltiples seeds y agregar resultados

Pipeline de agregación:
- [../scripts/aggregate_results.py](../scripts/aggregate_results.py)
- [../scripts/generate_aggregate_report.py](../scripts/generate_aggregate_report.py)

---

## 11) Cierre y recomendaciones

Orden práctico sugerido para baseline:
1. Empezar por PPO.
2. Usar A2C para un baseline actor-critic más simple.
3. Usar A3C cuando interese paralelismo CPU en recolección.
4. Usar TRPO si se necesita control más conservador del update y hay presupuesto de cómputo.

Para bases conceptuales sólidas, siempre anclar primero en Parte 1 (policy optimization + Monte Carlo + REINFORCE) antes del deep-dive de variantes.

---

## 12) Fuentes sugeridas usadas en la narrativa de Parte 2

- A2C: https://huggingface.co/blog/deep-rl-a2c
- A3C: https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
- PPO: https://huggingface.co/blog/deep-rl-ppo
- PPO (patrones de implementación): https://docs.pytorch.org/rl/0.7/tutorials/multiagent_ppo.html
- TRPO: https://dilithjay.com/blog/trpo
- TRPO: https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9
- TRPO: https://towardsdatascience.com/trust-region-policy-optimization-trpo-explained-4b56bd206fc2/