# DiMε similarity-threshold sensitivity analysis

All runs use the existing analytical relation, the complete eight-attribute DiMε lattice, greedy `g3`, `g3 <= 0.10`, and 30 seeded RHS permutations.

The purpose of this analysis is not to find an optimal threshold set. It is a
robustness check used to qualify the interpretation of the main `medium` run,
which remains the configuration reported as the primary DiMε result.

## Summary

| Configuration | RFDs | Distinct RHS | WSPM share | Best rule | g3 | Support | Confidence | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_075x | 19 | 6 | 0.632 | time_slot, NO2, O3, TEMP -> WSPM | 0.095 | 0.00157 | 0.894 | 1.335 |
| medium | 5 | 1 | 1.000 | station, PM10, NO2, O3, TEMP -> WSPM | 0.092 | 0.00161 | 0.899 | 1.163 |
| loose_150x | 1 | 1 | 1.000 | ∅ -> WSPM | 0.024 | 1.000 | 0.891 | 1.000 |
| wspm_strict_05 | 4 | 1 | 1.000 | station, time_slot, PM10, NO2, O3, TEMP -> WSPM | 0.087 | 0.00066 | 0.796 | 1.512 |
| wspm_loose_15 | 1 | 1 | 1.000 | ∅ -> WSPM | 0.024 | 1.000 | 0.891 | 1.000 |
| wspm_loose_20 | 1 | 1 | 1.000 | ∅ -> WSPM | 0.007 | 1.000 | 0.953 | 1.000 |

## RHS analysis

- `strict_075x`: NO2, PM10, PM2.5, WSPM, station, time_slot
- `medium`: WSPM
- `loose_150x`: WSPM
- `wspm_strict_05`: WSPM
- `wspm_loose_15`: WSPM
- `wspm_loose_20`: WSPM

## WSPM dominance

- `strict_075x`: other RHS emerge (NO2, PM10, PM2.5, WSPM, station, time_slot).
- `medium`: WSPM is the only RHS (5/5 RFDs).
- `loose_150x`: WSPM is the only RHS (1/1 RFDs).
- `wspm_strict_05`: WSPM is the only RHS (4/4 RFDs).
- `wspm_loose_15`: WSPM is the only RHS (1/1 RFDs).
- `wspm_loose_20`: WSPM is the only RHS (1/1 RFDs).

## Interpretation

- `medium` produces 5 RFDs and all of them have RHS `WSPM`.
- `strict_075x` expands the result to 19 RFDs and 6 distinct RHS, so WSPM dominance is not an absolute dataset property.
- Even under `strict_075x`, WSPM remains the most frequent RHS with 12 rules out of 19.
- Looser thresholds, and overly loose WSPM-specific thresholds, collapse to `∅ -> WSPM`; these rules have support `1` and lift `1`, so they are effectively uninformative.
- `medium` remains the main configuration because it balances compactness and interpretability without either exploding into many local rules or collapsing into trivial ones.

The UDT framing is conceptual. RFDs are approximate regularities, not causal laws; violations may reflect anomalies, data-quality issues, or unobserved events. Pairwise validation is quadratic, so the established analytical relation is intentionally reduced.
