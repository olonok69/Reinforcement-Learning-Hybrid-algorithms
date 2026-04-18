# RL Continuous Control — Guía del Presentador (60 minutos)

> **Sesión 6 del curso de RL.** Las sesiones anteriores cubrieron MDPs, Q-learning, DQN,
> policy gradient y A2C/PPO. La audiencia está familiarizada con value functions, redes neuronales
> y el policy-gradient theorem. Construye sobre esa base — no es necesario re-explicar las
> ecuaciones de Bellman desde cero.

---

## Nota sobre el título

El slide deck está titulado **"Hybrid Algorithms, the best from both worlds: DDPG, SAC & TD3"**.

El subtítulo *"best from both worlds"* es conceptualmente correcto: los tres algoritmos son
métodos **actor-critic** que combinan un **actor de policy gradient** (mundo 1) con un
**critic basado en valor** (mundo 2). Vale la pena hacer explícito ese punto en el slide de apertura.

Sin embargo, el término *"Hybrid"* como etiqueta principal no es estándar. En la literatura de RL
"hybrid" generalmente implica combinar RL model-free con planificación model-based, o RL con
aprendizaje supervisado. Un título más preciso y encontrable sería:

> **RL Continuous Control: Off-Policy Actor-Critic Methods — DDPG, TD3 & SAC**
> *Combinando lo mejor del policy gradient y el aprendizaje basado en valor*

**Recomendación:** conservar el tagline "best from both worlds" como subtítulo, pero encabezar
con "Continuous Control" u "Off-Policy Actor-Critic" para que la audiencia lo ubique
inmediatamente en el clúster correcto de la literatura. Si el curso utiliza "Hybrid Algorithms"
como tema recurrente para actor-critic, es una convención válida dentro del curso — simplemente
decláralo explícitamente en el primer minuto.

---

## Alcance

Tres algoritmos actor-critic off-policy para **espacios de acción continuos**:

| Algoritmo | Idea clave | Año |
|-----------|------------|-----|
| DDPG | Actor determinista + critic único + target nets | Lillicrap et al. 2015 |
| TD3  | Twin critics + delayed actor + target smoothing | Fujimoto et al. 2018 |
| SAC  | Actor estocástico + regularización por entropía + twin critics | Haarnoja et al. 2018 |

Estructura del repositorio usada en esta sesión:

```
rl_continuous_optimization/
├── benchmarks/          ← implementaciones de todos los algoritmos + replay buffer
│   ├── ddpg.py          ← DDPG: Actor, Critic, run_ddpg()
│   ├── td3.py           ← TD3:  Actor, TwinCritic, run_td3()
│   ├── sac.py           ← SAC:  GaussianActor, TwinCritic, run_sac()
│   ├── replay_buffer.py ← store de experiencia off-policy compartido
│   └── common.py        ← BenchmarkResult, run_timed(), métricas
├── ddpg_benchmark.py    ← CLI standalone para DDPG
├── td3_benchmark.py     ← CLI standalone para TD3
├── sac_benchmark.py     ← CLI standalone para SAC
├── run_all_comparison.py← runner unificado multi-algoritmo
└── scripts/
    ├── aggregate_results.py      ← calcular mean/std entre seeds
    └── generate_aggregate_report.py ← generar reporte HTML/texto
```

Documentos conceptuales: [`01_ddpg.md`](es/01_ddpg.md) · [`02_td3.md`](es/02_td3.md) · [`03_sac.md`](es/03_sac.md)  
Guía de código: [`../benchmarks/README.md`](../benchmarks/README.md)

---

## Objetivos de aprendizaje

Al terminar la sesión, la audiencia debe ser capaz de:

1. Explicar por qué `argmax_a Q(s,a)` estilo DQN no escala en espacios de acción continuos.
2. Trazar el loop completo de DDPG: actor genera acción → critic evalúa → gradiente vuelve al actor.
3. Nombrar las tres correcciones de TD3 y explicar *por qué* cada una mejora la estabilidad.
4. Explicar qué hace el término de entropía en SAC y cómo `alpha` controla el tradeoff
   exploración-explotación.
5. Leer el código del benchmark e identificar las líneas exactas que implementan cada idea clave.
6. Ejecutar una comparación reproducible de los tres algoritmos e interpretar las métricas de salida.

---

## Flujo detallado de 60 minutos

---

### 0–5 min — El problema de las acciones continuas y el límite de DQN

**Focus del slide:** motivar por qué se necesita una nueva familia de algoritmos.

**Puntos a desarrollar:**

- DQN selecciona acciones mediante `argmax_a Q(s,a)`. Ese `argmax` se calcula por enumeración
  o un forward pass sobre una cabeza de salida discreta. Con un brazo robótico de 7 articulaciones
  que se mueve en `[-1, 1]` el espacio de búsqueda es infinitamente continuo — la enumeración
  es imposible.
- Discretizar un espacio de acción continuo pierde precisión y explota combinatoriamente:
  10 articulaciones × 100 bins = 10^20 combinaciones.
- La solución introducida en DDPG: **aprender el argmax directamente** como función paramétrica
  independiente (el actor `mu(s; theta)`). Esa es la idea central del actor-critic aplicada al
  control continuo.
- Conectar con sesiones anteriores: "En PPO usamos policy gradient para aprender una policy
  *estocástica*. DDPG aprende una policy *determinista* y la evalúa con un critic tipo Q-function."

**Pregunta anticipada:** *"¿Por qué no usar PPO con una cabeza Gaussiana para acciones continuas?"*  
Respuesta: Se puede (y funciona). DDPG/TD3/SAC son **off-policy** — reutilizan experiencia pasada
mediante un replay buffer, lo que los hace más eficientes en muestras para simuladores lentos o
entornos de robot real. PPO es on-policy y descarta los datos anteriores.

---

### 5–20 min — DDPG: el blueprint actor-critic para control continuo

**Focus del slide:** los cuatro componentes de DDPG y cómo interactúan.

#### 5–9 min — Arquitectura general

**Puntos a desarrollar:**

- Cuatro redes, siempre en dos pares:
  - **Actor online** `mu(s; theta)` + **actor target** `mu(s; theta')`  
  - **Critic online** `Q(s,a; phi)` + **critic target** `Q(s,a; phi')`
- Las target networks existen únicamente para estabilizar el objetivo de Bellman — evolucionan
  lentamente mediante soft update (Polyak). Sin ellas la red persigue un objetivo en movimiento
  con sus propios gradientes y el entrenamiento diverge.
- El **replay buffer** desacopla la recolección de datos del aprendizaje: cualquier transición
  pasada `(s,a,r,s')` puede muestrearse, rompiendo la correlación temporal que invalida SGD.

**Referencia de código — definición de redes:**  
[`benchmarks/ddpg.py` líneas 35–63](../benchmarks/ddpg.py)

```python
# Actor: state -> tanh(net(state)) * max_action
# tanh comprime la salida a (-1,1); max_action reescala a los límites del entorno
class Actor(nn.Module):
    def forward(self, state):
        return torch.tanh(self.net(state)) * self.max_action

# Critic: concatenar (state, action) -> Q-value escalar
class Critic(nn.Module):
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))
```

Señalar: el critic toma `(s, a)` como entrada conjunta. Esto sólo es posible porque durante
el entrenamiento lo consultamos con la acción elegida por el actor — no sobre todas las acciones.

**Referencia de código — instanciación de redes y copia a targets:**  
[`benchmarks/ddpg.py` líneas 77–83](../benchmarks/ddpg.py)

```python
actor_target.load_state_dict(actor.state_dict())   # comienzan idénticos
critic_target.load_state_dict(critic.state_dict()) # comienzan idénticos
```

#### 9–14 min — Warmup y exploración

**Puntos a desarrollar:**

- Los algoritmos off-policy necesitan el buffer pre-cargado antes del primer paso de gradiente.
  Los primeros `warmup_steps=5000` pasos de entorno usan **acciones aleatorias** para sembrar
  experiencia diversa.
- Durante el entrenamiento, la exploración es ruido Gaussiano aditivo sobre la acción determinista:
  `a = clip(mu(s) + N(0, sigma), low, high)`  
  DDPG no tiene mecanismo de exploración intrínseco — el diseñador elige `sigma`.
- Esta es una debilidad conocida: si el ruido está mal calibrado, el agente sobre-explora o
  queda atrapado en un óptimo local.

**Referencia de código — warmup y ruido:**  
[`benchmarks/ddpg.py` líneas 101–108](../benchmarks/ddpg.py)

```python
if global_step < cfg.warmup_steps:
    action = env.action_space.sample()       # totalmente aleatorio
else:
    action = actor(state_t).cpu().numpy()    # actor determinista
    noise = np.random.normal(0.0, cfg.exploration_noise * max_action, ...)
    action = np.clip(action + noise, ...)    # limitar a los bounds del entorno
```

Configuración: [`DDPGConfig`](../benchmarks/ddpg.py) — `warmup_steps=5000`,
`exploration_noise=0.1`.

#### 14–20 min — Las ecuaciones de actualización

**Puntos a desarrollar:**

Seguir la actualización en el orden en que ejecuta en el código:

1. **Muestrear un mini-batch** `(s, a, r, s', done)` del replay buffer.
2. **Calcular el objetivo de Bellman** (con target networks, sin gradiente):
   $$y = r + \gamma (1 - \text{done}) \cdot Q_{\text{target}}(s', \mu_{\text{target}}(s'))$$
3. **Actualizar el critic** — minimizar MSE entre la estimación Q online y el objetivo:
   $$L_{\text{critic}} = \text{MSE}(Q(s,a),\ y)$$
4. **Actualizar el actor** — maximizar Q mediante gradient ascent, implementado como minimizar el negativo:
   $$L_{\text{actor}} = -\frac{1}{N}\sum Q(s,\ \mu(s))$$
5. **Soft update** de ambas target networks (Polyak averaging, `tau=0.005`):
   $$\theta' \leftarrow (1-\tau)\theta' + \tau\theta$$

**Referencia de código — bloque de actualización completo:**  
[`benchmarks/ddpg.py` líneas 118–144](../benchmarks/ddpg.py)

```python
# 2-3: actualización del critic
next_action = actor_target(ns)
target_q = critic_target(ns, next_action)
target_q = r + nd * cfg.gamma * target_q      # nd = (1 - done)
critic_loss = F.mse_loss(critic(s, a), target_q)
critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

# 4: actualización del actor — gradiente fluye a través del critic hacia el actor
actor_loss = -critic(s, actor(s)).mean()
actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

# 5: Polyak soft update de ambos pares target
for p, tp in zip(critic.parameters(), critic_target.parameters()):
    tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)
```

**Insight clave a destacar:** el actor se actualiza *diferenciando a través* del critic.
La cadena de gradiente es: `actor_loss → Q(s,a) → actor(s) → actor.parameters()`. Este es el
deterministic policy gradient theorem (Silver et al. 2014).

**Documento conceptual:** [`es/01_ddpg.md`](es/01_ddpg.md)  
**Riesgos conocidos:** critic único → se acumula sobreestimación de Q → el actor persigue valores
inflados → inestabilidad. Esto motiva TD3.

---

### 20–35 min — TD3: tres correcciones quirúrgicas a DDPG

**Focus del slide:** mostrar el código de DDPG, luego el diff contra TD3 para hacer concretas las mejoras.

**Puntos introductorios:**  
Fujimoto et al. 2018 diagnosticó tres modos de falla en DDPG y añadió una corrección a cada uno.
Los cambios son pequeños en código pero grandes en impacto.

**Documento conceptual:** [`es/02_td3.md`](es/02_td3.md)

#### Corrección 1 — Twin critics (líneas 47–73 de td3.py)

**Puntos a desarrollar:**

- El critic único de DDPG sobreestima Q porque el actor siempre empuja hacia el máximo de Q.
  Con ruido y errores de aproximación, ese sesgo se acumula.
- TD3 mantiene **dos critics independientes** `Q1` y `Q2`. El objetivo de Bellman usa su
  **mínimo**, no ninguno de los dos por separado:
  $$y = r + \gamma(1-\text{done})\cdot\min(Q_1^{\text{target}}(s',a'),\ Q_2^{\text{target}}(s',a'))$$
- Min-de-dos es un estimador pesimista, pero el pesimismo es más seguro que el optimismo cuando
  el actor explota continuamente los errores del critic.

**Referencia de código — TwinCritic y clipped double-Q target:**  
[`benchmarks/td3.py` líneas 47–73](../benchmarks/td3.py) y
[líneas 119–121](../benchmarks/td3.py)

```python
class TwinCritic(nn.Module):
    # q1 y q2 son MLPs independientes, misma arquitectura que el Critic de DDPG
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
    def q1_only(self, state, action):     # usado en la actualización del actor
        return self.q1(torch.cat([state, action], dim=1))

# target: mínimo de los dos critics
target_q1, target_q2 = critic_target(ns, next_action)
target_q = r + nd * cfg.gamma * torch.min(target_q1, target_q2)  # clipped!
```

#### Corrección 2 — Delayed actor updates (`policy_delay`)

**Puntos a desarrollar:**

- En DDPG el actor y el critic se actualizan cada paso. Pero si el critic es impreciso (inicio
  del entrenamiento), el gradiente del actor es ruidoso y puede empujar la policy en dirección
  equivocada.
- TD3 actualiza el actor sólo cada `policy_delay=2` actualizaciones del critic, dando tiempo al
  critic para converger primero.
- Importante: cuando el actor sí se actualiza, las target networks se actualizan al mismo tiempo.
  Esto mantiene el sistema sincronizado.

**Referencia de código — gate del actor retrasado:**  
[`benchmarks/td3.py` líneas 122–133](../benchmarks/td3.py)

```python
update_step += 1

# critics se actualizan cada paso
critic_loss = ...
critic_opt.step()

# actor (y targets) sólo se actualiza cada policy_delay pasos
if update_step % cfg.policy_delay == 0:
    actor_loss = -critic.q1_only(s, actor(s)).mean()
    actor_opt.step()
    # soft update de targets aquí (dentro del if)
```

#### Corrección 3 — Target policy smoothing

**Puntos a desarrollar:**

- Sin smoothing, el actor target siempre produce la misma acción determinista para un `s'` dado.
  Si el critic tiene un pico agudo en esa acción, el Q target es artificialmente elevado.
- TD3 añade **ruido Gaussiano recortado** a la acción target, promediando efectivamente el Q target
  sobre una pequeña vecindad de acciones — regularizando la superficie de Q:
  $$a' = \text{clip}\!\left(\mu_{\text{target}}(s') + \epsilon,\ -a_{\max},\ a_{\max}\right), \quad
    \epsilon = \text{clip}\!\left(\mathcal{N}(0,\sigma),\ -c,\ c\right)$$

**Valores de configuración:** `policy_noise=0.2`, `noise_clip=0.5`.

**Referencia de código — ruido de target smoothing:**  
[`benchmarks/td3.py` líneas 113–121](../benchmarks/td3.py)

```python
noise = (torch.randn_like(a) * cfg.policy_noise * max_action).clamp(
    -cfg.noise_clip * max_action,
    +cfg.noise_clip * max_action,
)
next_action = (actor_target(ns) + noise).clamp(-max_action, max_action)
```

**Tabla resumen para el slide:**

| Problema en DDPG | Corrección TD3 | Parámetro |
|---|---|---|
| Sobreestimación de Q | Twin critics, usar mínimo | *(estructural)* |
| Actor persigue critic ruidoso | Retrasar actualizaciones del actor | `policy_delay=2` |
| Picos agudos de Q en acción target | Ruido sobre acción target | `policy_noise`, `noise_clip` |

**Parámetros extra de TD3Config vs DDPGConfig:**  
[`benchmarks/td3.py` líneas 26–30](../benchmarks/td3.py) — `policy_noise`, `noise_clip`, `policy_delay`.

---

### 35–50 min — SAC: exploración con principios mediante maximización de entropía

**Focus del slide:** el objetivo aumentado con entropía y por qué cambia todo el diseño de la policy.

**Puntos introductorios:**  
DDPG y TD3 exploran a través de ruido externo añadido a una policy determinista. SAC hace la
exploración *intrínseca*: la policy es estocástica y recibe una recompensa explícita por tener
alta entropía. Esto cambia la arquitectura del actor de un mapeo determinista a una distribución
Gaussiana.

**Documento conceptual:** [`es/03_sac.md`](es/03_sac.md)

#### 35–40 min — El objetivo con regularización de entropía

**Puntos a desarrollar:**

RL estándar maximiza retorno acumulado:
$$J(\pi) = \mathbb{E}\!\left[\sum_t \gamma^t r_t\right]$$

SAC aumenta esto con la entropía de la policy $\mathcal{H}(\pi(\cdot|s))$, controlada por la temperatura `alpha`:
$$J_{\text{SAC}}(\pi) = \mathbb{E}\!\left[\sum_t \gamma^t \bigl(r_t + \alpha\,\mathcal{H}(\pi(\cdot|s_t))\bigr)\right]$$

Efecto de `alpha`:
- `alpha` alto → fuerte presión a ser estocástico → más exploración, convergencia más lenta.
- `alpha` bajo → principalmente orientado a la recompensa → se aproxima a TD3/DDPG.
- `alpha=0` → equivale a una policy determinista dura (similar a DDPG pero con twin critics).

El término de entropía se propaga al **objetivo de Bellman** del critic:
$$y = r + \gamma(1-\text{done})\left[\min(Q_1^{\text{target}},Q_2^{\text{target}})(s',a') - \alpha\log\pi(a'|s')\right]$$

Y al **actor loss**:
$$L_{\text{actor}} = \mathbb{E}_{a\sim\pi}\!\left[\alpha\log\pi(a|s) - \min(Q_1,Q_2)(s,a)\right]$$

#### 40–47 min — GaussianActor: sampling estocástico con tanh squashing

**Puntos a desarrollar:**

- El actor produce una **distribución Gaussiana** `N(mu(s), sigma(s))`, no una acción única.  
  Dos cabezas de salida: `mean` y `log_std` (acotado a `[LOG_STD_MIN, LOG_STD_MAX] = [-20, 2]`).
- La acción se muestrea mediante reparametrización: `z ~ N(mu, sigma)` y luego squashing:
  `action = tanh(z) * max_action`  
  La reparametrización (`rsample`) hace la muestra diferenciable, permitiendo que los gradientes
  fluyan a través de ella.
- **Corrección del log-probability por tanh squashing:**  
  $$\log\pi(a|s) = \log\mathcal{N}(z|\mu,\sigma) - \sum_i\log(1 - \tanh^2(z_i) + \epsilon)$$
  Esta es la fórmula de cambio de variables. Sin ella, `log_prob` sería incorrecto y la
  estimación de entropía estaría equivocada. El `1e-6` protege contra `log(0)`.

**Referencia de código — GaussianActor:**  
[`benchmarks/sac.py` líneas 38–73](../benchmarks/sac.py)

```python
class GaussianActor(nn.Module):
    def forward(self, state):
        h = self.backbone(state)
        mean    = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self(state)
        std  = log_std.exp()
        z    = mean if deterministic else Normal(mean, std).rsample()
        squashed = torch.tanh(z)
        action   = squashed * self.max_action

        # corrección de log-prob por tanh (cambio de variables)
        log_prob = Normal(mean, std).log_prob(z)
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = (log_prob - correction).sum(dim=1, keepdim=True)

        return action, log_prob
```

**Punto clave:** en tiempo de **inferencia/video**, `deterministic=True` da `z = mean` y
`log_prob = 0`. La policy colapsa a su modo determinista, útil para evaluación.

#### 47–50 min — Loop de actualización de SAC

**Puntos a desarrollar:**

- **Sin actor target network.** SAC no la necesita: la policy estocástica suaviza naturalmente
  la superficie target (el rol que cumple el target-policy noise en TD3).
- El **critic target** sí existe (copia de evolución lenta del twin critic).
- Orden de actualización por paso: critic → actor → soft update del critic target.

**Referencia de código — bloque de actualización SAC:**  
[`benchmarks/sac.py` líneas ~118–160](../benchmarks/sac.py)

```python
# Actualización del critic — nótese el término de entropía (- alpha * next_logp) en el target
next_a, next_logp = actor.sample(ns)
target_q = torch.min(*critic_target(ns, next_a)) - cfg.alpha * next_logp
target = r + nd * cfg.gamma * target_q
critic_loss = mse(critic(s,a)[0], target) + mse(critic(s,a)[1], target)

# Actualización del actor — minimizar (alpha*logp - min_Q)
pi_action, logp_pi = actor.sample(s)
min_q_pi = torch.min(*critic(s, pi_action))
actor_loss = (cfg.alpha * logp_pi - min_q_pi).mean()
```

**Diferencia estructural vs DDPG/TD3 para destacar en slide:**

| | DDPG | TD3 | SAC |
|---|---|---|---|
| Tipo de policy | Determinista | Determinista | Gaussiana estocástica |
| Critics | 1 | 2 (min) | 2 (min) |
| Actor target | Sí | Sí | **No** |
| Exploración | Ruido Gaussiano externo | Ruido Gaussiano externo | **Intrínseca (entropía)** |
| Entropía en target | No | No | **Sí** |
| Hiperparámetro extra | `exploration_noise` | `policy_noise`, `policy_delay` | `alpha` |

---

### 50–57 min — Comparación lado a lado y recomendaciones prácticas

**Focus del slide:** ejecutar el benchmark, leer la tabla de resultados, tomar una decisión.

#### Demo en vivo (ejecutar durante la sesión si hay tiempo, o grabado previamente)

Ejecutar desde `rl_continuous_optimization/`:

```bash
# Smoke test rápido — los tres algoritmos, 160 episodios cada uno
uv run python run_all_comparison.py \
    --methods ddpg td3 sac \
    --episodes 160 \
    --seed 42 \
    --output-dir outputs/session6
```

El script llama a cada algoritmo secuencialmente, recoge objetos `BenchmarkResult`
(definidos en [`benchmarks/common.py` líneas 19–29](../benchmarks/common.py)), e imprime una
tabla resumen con:

- `max_avg_reward_100` — mejor promedio móvil de 100 episodios observado (rendimiento pico)
- `final_avg_reward_100` — últimos 100 episodios (estabilidad de convergencia)
- `elapsed_sec` — tiempo de reloj de pared

**Forma esperada del output** (Pendulum-v1, seed 42):

```
Algoritmo  | Max Avg Reward (100ep) | Final Avg Reward (100ep) | Tiempo (s)
-----------|------------------------|--------------------------|----------
DDPG       | ~-200 a -150           | -200 a -160              | ~30s
TD3        | ~-150 a -100           | ~-140 a -110             | ~35s
SAC        | ~-120 a -80            | ~-120 a -90              | ~40s
```

> Rango de reward de Pendulum-v1: –1600 (peor) a ~–120 (casi óptimo).
> Con sólo 160 episodios la convergencia es parcial — esto es intencional para velocidad de demo.
> Las ejecuciones de producción usan 500–1000 episodios.

**Agregar resultados entre seeds:**

```bash
# Ejecutar con múltiples seeds
uv run python run_all_comparison.py --seed 0  --output-dir outputs/seed0
uv run python run_all_comparison.py --seed 1  --output-dir outputs/seed1
uv run python run_all_comparison.py --seed 42 --output-dir outputs/seed42

# Estadísticas agregadas
uv run python scripts/aggregate_results.py \
    --inputs outputs/seed*/comparison_results.json \
    --output-json outputs/aggregate_summary.json \
    --output-csv  outputs/aggregate_summary.csv

# Generar reporte HTML
uv run python scripts/generate_aggregate_report.py \
    --input  outputs/aggregate_summary.json \
    --output-dir outputs/report \
    --title "RL Continuous Control Aggregate Report"
```

El script [`scripts/aggregate_results.py`](../scripts/aggregate_results.py) calcula `mean ± std`
por algoritmo entre seeds. Es la única forma estadísticamente válida de comparar algoritmos de RL
— los números de una sola seed son poco fiables.

#### Slide de recomendaciones prácticas

| Escenario | Elección | Razón |
|---|---|---|
| Primer experimento en una tarea nueva | **SAC** | Robusto, exploración incorporada, poco tuning |
| Se necesita policy determinista en producción | **TD3** | Salida determinista, estimaciones Q estables |
| Enseñanza / baseline de ablation | **DDPG** | Código más simple, todos los problemas visibles |
| Mayor velocidad de wall-clock por muestra | **DDPG** | Critic único, sin cálculo de log-prob |
| Mejor eficiencia de muestras esperada | **SAC o TD3** | Ambos significativamente mejores que DDPG |

Regla general: **comenzar con SAC, comparar con TD3, mantener DDPG como ancla pedagógica**.

---

### 57–60 min — Guía de Q&A

Preguntas anticipadas y respuestas cortas:

**P: ¿Podemos auto-ajustar `alpha` en SAC en lugar de fijarlo?**  
R: Sí. El ajuste automático de entropía trata `alpha` como multiplicador de Lagrange apuntando
a un nivel de entropía deseado. Añade un optimizador Adam para `log_alpha`. No está implementado
en este benchmark (usa `alpha=0.2` fijo), pero es común en SAC de producción. Referencia:
Haarnoja et al. 2018 (v2).

**P: ¿Por qué SAC es más lento por paso que DDPG?**  
R: Cada actualización de SAC requiere el cálculo de `Normal.log_prob` y su gradiente, más la
corrección por tanh. Es barato pero no gratuito. El tradeoff generalmente vale la pena.

**P: ¿Funciona SAC para espacios de acción discretos?**  
R: El actor Gaussiano es específico para espacios continuos. El SAC discreto usa una policy
softmax y reemplaza la reparametrización con una expectativa directa sobre todas las acciones.
No se cubre aquí.

**P: ¿Cuándo ayuda más el ruido de exploración que la regularización por entropía?**  
R: El ruido de exploración (DDPG/TD3) puede moldearse (Ornstein-Uhlenbeck, decay programado,
conocimiento del dominio). La regularización por entropía (SAC) es más automática pero menos
personalizable. Para espacios de acción muy restringidos o entornos safety-critical, el ruido
explícito suele ser más seguro.

**P: ¿El replay buffer es el mismo para los tres algoritmos?**  
R: Sí — ver [`benchmarks/replay_buffer.py`](../benchmarks/replay_buffer.py). Misma implementación
de buffer circular con almacenamiento fijo de `not_done`. El buffer es agnóstico al algoritmo.

---

## Mapa completo concepto–código

| Concepto | Documento | Archivo de código | Líneas clave |
|---|---|---|---|
| Actor determinista (tanh) | [`es/01_ddpg.md`](es/01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 35–48 |
| Critic único (state+action) | [`es/01_ddpg.md`](es/01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 51–63 |
| Loop de entrenamiento DDPG | [`es/01_ddpg.md`](es/01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 66–175 |
| Warmup + ruido de exploración | [`es/01_ddpg.md`](es/01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 101–108 |
| Actualiz. critic + actor + Polyak | [`es/01_ddpg.md`](es/01_ddpg.md) | [`benchmarks/ddpg.py`](../benchmarks/ddpg.py) | 118–144 |
| Twin critic (Q1+Q2) | [`es/02_td3.md`](es/02_td3.md) | [`benchmarks/td3.py`](../benchmarks/td3.py) | 47–73 |
| Clipped double-Q target | [`es/02_td3.md`](es/02_td3.md) | [`benchmarks/td3.py`](../benchmarks/td3.py) | 119–121 |
| Target policy smoothing | [`es/02_td3.md`](es/02_td3.md) | [`benchmarks/td3.py`](../benchmarks/td3.py) | 113–118 |
| Delayed actor update | [`es/02_td3.md`](es/02_td3.md) | [`benchmarks/td3.py`](../benchmarks/td3.py) | 122–133 |
| GaussianActor (mean+log_std) | [`es/03_sac.md`](es/03_sac.md) | [`benchmarks/sac.py`](../benchmarks/sac.py) | 38–55 |
| Reparametrización + tanh squash | [`es/03_sac.md`](es/03_sac.md) | [`benchmarks/sac.py`](../benchmarks/sac.py) | 56–73 |
| SAC critic target (con entropía) | [`es/03_sac.md`](es/03_sac.md) | [`benchmarks/sac.py`](../benchmarks/sac.py) | 147–153 |
| SAC actor loss (con entropía) | [`es/03_sac.md`](es/03_sac.md) | [`benchmarks/sac.py`](../benchmarks/sac.py) | 155–161 |
| Replay buffer | todos | [`benchmarks/replay_buffer.py`](../benchmarks/replay_buffer.py) | 1–43 |
| BenchmarkResult + métricas | todos | [`benchmarks/common.py`](../benchmarks/common.py) | 19–47 |
| Runner CLI multi-método | todos | [`run_all_comparison.py`](../run_all_comparison.py) | 1–end |
| Control de seed | todos | [`benchmarks/common.py`](../benchmarks/common.py) | 31–34 |

---

## Referencias

### DDPG
- Lillicrap et al. (2015) — paper original: https://arxiv.org/abs/1509.02971
- Explicado: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-4643c1f71b2e/
- SpinningUp: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
- Coach: https://intellabs.github.io/coach/components/agents/policy_optimization/ddpg.html
- Aplicación reciente: https://www.nature.com/articles/s41598-025-99213-3

### TD3
- Fujimoto et al. (2018) — paper original: https://arxiv.org/abs/1802.09477
- Explicado: https://medium.com/@heyamit10/twin-delayed-deep-deterministic-policy-gradient-td3-fc8e9950f029
- SpinningUp: https://spinningup.openai.com/en/latest/algorithms/td3.html

### SAC
- Haarnoja et al. (2018) v1: https://arxiv.org/abs/1801.01290
- Haarnoja et al. (2018) v2 (auto-alpha): https://arxiv.org/abs/1812.05905
- Explicado: https://towardsdatascience.com/navigating-soft-actor-critic-reinforcement-learning-8e1a7406ce48/
- SpinningUp: https://spinningup.openai.com/en/latest/algorithms/sac.html
