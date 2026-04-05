# Documentación de Algoritmos (Español)

Esta carpeta contiene versiones en español de los documentos de algoritmos usados en el benchmark.

## Documentos
- `../presentation_guide_60min_es.md` (Parte 1: policy optimization + Monte Carlo + REINFORCE)
- `../presentation_guide_part2_60min_es.md` (Parte 2: A2C, A3C, PPO, TRPO)
- `../policy_gradient_line_by_line.md` (walkthrough detallado en inglés por línea de `benchmarks/policy_gradient.py`)
- `../policy_gradient_line_by_line_es.md` (walkthrough detallado en español por línea)
- `03_policy_gradient.md`
- `04_a2c.md`
- `05_a3c.md`
- `06_ppo.md`
- `07_trpo.md`

## Protocolo recomendado de experimentos
1. Ejecutar cada algoritmo con múltiples seeds (mínimo 3).
2. Mantener budgets de entrenamiento comparables entre métodos.
3. Reportar tendencia central y varianza (media ± std).
4. Guardar salidas crudas y logs de error para reproducibilidad.
5. Documentar caveats de cada método (por ejemplo, dependencia de TRPO o escalado de CPU en A3C).
