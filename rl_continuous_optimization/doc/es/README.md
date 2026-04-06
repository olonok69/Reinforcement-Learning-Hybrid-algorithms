# Documentacion de Algoritmos (Control Continuo)

Esta carpeta contiene versiones en espanol de la documentacion por algoritmo para `rl_continuous_optimization`.

## Documentos
- `../presentation_guide_60min.md`
- `../presentation_guide_60min_es.md`
- `01_ddpg.md`
- `02_td3.md`
- `03_sac.md`

## Enlaces utiles
- Indice principal: `../README.md`
- Runner unificado: `../../run_all_comparison.py`
- Scripts de agregacion: `../../scripts/aggregate_results.py`
- Script de reporte: `../../scripts/generate_aggregate_report.py`

## Protocolo recomendado
1. Ejecutar cada metodo con minimo 3 seeds.
2. Mantener mismo entorno y budget entre metodos.
3. Agregar resultados con los scripts de `../../scripts`.
4. Revisar `comparison_errors.json` antes de conclusiones.
