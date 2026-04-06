# Continuous Control Algorithm Documentation

This folder contains per-algorithm documentation for the continuous-action benchmark suite in `rl_continuous_optimization`.

## Documents
- `presentation_guide_60min.md`
- `presentation_guide_60min_es.md`
- `01_ddpg.md`
- `02_td3.md`
- `03_sac.md`

## Spanish versions
- `es/README.md`
- `es/01_ddpg.md`
- `es/02_td3.md`
- `es/03_sac.md`

## Implementation anchors
- Shared benchmark helpers: `../benchmarks/common.py`
- Replay buffer: `../benchmarks/replay_buffer.py`
- Unified runner: `../run_all_comparison.py`

## Recommended comparison protocol
1. Run each algorithm with at least 3 seeds.
2. Keep environment and budget aligned across methods.
3. Aggregate results with `../scripts/aggregate_results.py`.
4. Generate report artifacts with `../scripts/generate_aggregate_report.py`.
5. Keep `comparison_errors.json` in your final analysis.
