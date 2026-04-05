# Policy Gradient (REINFORCE)

## 1) Qué resuelve REINFORCE
REINFORCE es un método de policy gradient que **optimiza la policy directamente** en lugar de aprender primero una tabla de valor.

Dada una policy estocástica $\pi_\theta(a|s)$, maximizamos el retorno esperado:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

El estimador clave es:

$$
\nabla_\theta J(\theta) \propto \mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\,G_t\right]
$$

Donde $G_t$ es el retorno descontado desde el paso temporal $t$.

---

## 2) Intuición (paso a paso)
REINFORCE sigue un patrón Monte Carlo:
1. Ejecutar un episodio completo con la policy estocástica actual.
2. Calcular retornos descontados para cada paso.
3. Aumentar la probabilidad de acciones que llevaron a alto retorno.
4. Reducir la probabilidad de acciones que llevaron a bajo retorno.
5. Repetir durante muchos episodios.

Es un método simple y limpio conceptualmente, pero con varianza alta por usar retornos de episodio completo.

---

## 3) Por qué se usan log-probabilities
El update usa $\log\pi_\theta(a_t|s_t)$ (no probabilidades crudas) porque:
- Mejora estabilidad numérica en el cálculo de gradientes.
- Aparece de forma natural con el log-derivative trick en policy gradients.
- Facilita el manejo de productos de probabilidades de trayectoria durante la optimización.

En código, esto aparece como `dist.log_prob(action)`.

---

## 4) De dónde viene la varianza
REINFORCE usa retornos de episodio completo:

$$
G_t = r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\dots
$$

Incluso para estados similares, las trayectorias muestreadas pueden variar bastante, lo que causa oscilaciones en los gradientes.

Estrategias comunes de mitigación:
- Normalización de retornos
- Baselines / critics (familia actor-critic)
- Regularización por entropía (opcional) para evitar convergencia determinista temprana

---

## 5) Cómo implementa este repo REINFORCE
Hay dos implementaciones relevantes:

- Módulo benchmark: [benchmarks/policy_gradient.py](../../benchmarks/policy_gradient.py)
- Script demo/CLI: [Policy_Gradient/reinforce_cartpole.py](../../Policy_Gradient/reinforce_cartpole.py)

El script [Policy_Gradient/reinforce_cartpole.py](../../Policy_Gradient/reinforce_cartpole.py) es el más práctico para demo y grabación de video.

### Mapa de seguimiento de código (script)
- Definición de red de policy: [Policy_Gradient/reinforce_cartpole.py](../../Policy_Gradient/reinforce_cartpole.py#L13)
- Cálculo de retornos descontados + normalización: [Policy_Gradient/reinforce_cartpole.py](../../Policy_Gradient/reinforce_cartpole.py#L29)
- Bucle principal de entrenamiento (muestreo, trayectoria, update): [Policy_Gradient/reinforce_cartpole.py](../../Policy_Gradient/reinforce_cartpole.py#L41)
- Evaluación con render humano: [Policy_Gradient/reinforce_cartpole.py](../../Policy_Gradient/reinforce_cartpole.py#L111)
- Evaluación con grabación de video: [Policy_Gradient/reinforce_cartpole.py](../../Policy_Gradient/reinforce_cartpole.py#L139)
- Argumentos CLI: [Policy_Gradient/reinforce_cartpole.py](../../Policy_Gradient/reinforce_cartpole.py#L180)

### Update usado en este script
El script calcula retornos Monte Carlo normalizados y aplica:

$$
\mathcal{L}_{policy} = -\sum_t \log\pi_\theta(a_t|s_t)\,\hat{G}_t
$$

Donde $\hat{G}_t$ es el retorno normalizado.

---

## 6) Comandos CLI (probados)
Ejecutar desde la raíz del repositorio.

### Smoke test rápido
```bash
uv run python Policy_Gradient/reinforce_cartpole.py --episodes 5 --log-every 1 --seed 42
```

### Entrenamiento estándar
```bash
uv run python Policy_Gradient/reinforce_cartpole.py --episodes 1200 --gamma 0.99 --lr 1e-3 --log-every 25 --seed 42
```

### Grabar video demo después de entrenar
```bash
uv run python Policy_Gradient/reinforce_cartpole.py --episodes 300 --record-video --video-episodes 3 --video-dir outputs/videos/reinforce_cartpole_demo --seed 42
```

### Render + grabación para demo en vivo
```bash
uv run python Policy_Gradient/reinforce_cartpole.py --episodes 300 --record-and-render --render-episodes 2 --video-episodes 2 --video-dir outputs/videos/reinforce_cartpole_demo --seed 42
```

Los videos se guardan en la carpeta definida por `--video-dir`.

---

## 7) Checklist para demo (grabación)
1. Ejecutar un entrenamiento corto con seed fija.
2. Grabar al menos 1 episodio de evaluación con `--record-video`.
3. Verificar que existan archivos en `outputs/videos/reinforce_cartpole_demo`.
4. Opcional: usar `--render-eval` para presentación en vivo.

---

## 8) Fortalezas y limitaciones
### Fortalezas
- Lógica algorítmica muy clara.
- Funciona naturalmente con policies estocásticas.
- Base educativa excelente para actor-critic/PPO.

### Limitaciones
- Updates con alta varianza.
- Baja eficiencia muestral por ser on-policy.
- Sensible a learning rate, escala de retornos y seed.

---

## 9) Tips prácticos de tuning
- Si el entrenamiento es inestable, bajar `--lr`.
- Si converge lento o con ruido, aumentar episodios y comparar varias seeds.
- Mantener $\gamma$ cerca de $0.99$ en CartPole, salvo que quieras favorecer más la recompensa inmediata.
- Usar normalización de retornos (ya implementada en el script) para estabilizar gradientes.

---

## 10) Referencia usada para enriquecer
Esta página se enriqueció usando ideas del siguiente artículo (reformuladas y adaptadas a este codebase):
- https://shivang-ahd.medium.com/policy-gradient-methods-with-reinforce-a-step-by-step-guide-to-reinforcement-learning-mastery-51fe855a504f
