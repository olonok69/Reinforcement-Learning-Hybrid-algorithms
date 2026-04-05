# Algorithm Documentation

This folder contains detailed notes for every benchmark method implemented in this repository.

## Spanish versions
- Presentation guide Part 1 (ES): [presentation_guide_60min_es.md](presentation_guide_60min_es.md)
- Presentation guide Part 2 (ES): [presentation_guide_part2_60min_es.md](presentation_guide_part2_60min_es.md)
- Algorithm docs (ES): [es/README.md](es/README.md)

## Per-algorithm docs
- `presentation_guide_60min.md` (Part 1: detailed policy optimization + Monte Carlo + REINFORCE)
- `presentation_guide_part2_60min.md` (Part 2: A2C, A3C, PPO, TRPO)
- `policy_gradient_line_by_line.md` (deep line-by-line walkthrough of `benchmarks/policy_gradient.py`)
- `policy_gradient_line_by_line_es.md` (Spanish line-by-line companion)
- `03_policy_gradient.md`
- `04_a2c.md`
- `05_a3c.md`
- `06_ppo.md`
- `07_trpo.md`

## Recommended experiment protocol
When using these methods for comparison experiments:

1. Run each algorithm with multiple seeds (minimum 3).
2. Keep training budgets comparable across methods.
3. Report both central tendency and variance (mean ± std).
4. Store raw outputs and error logs for reproducibility.
5. Document any method-specific caveats (for example, TRPO dependency or A3C CPU scaling).
