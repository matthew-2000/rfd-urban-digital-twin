# DiMε similarity-threshold sensitivity analysis

All runs use the existing analytical relation, the complete eight-attribute DiMε lattice, greedy `g3`, `g3 <= 0.10`, and 30 seeded RHS permutations.

## Summary

| Configuration | RFDs | RHS | WSPM share | Best rule | g3 | Support | Confidence | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_075x | 19 | NO2, PM10, PM2.5, WSPM, station, time_slot | 0.632 | time_slot, NO2, O3, TEMP -> WSPM | 0.095 | 0.002 | 0.894 | 1.335 |
| medium | 5 | WSPM | 1.000 | station, PM10, NO2, O3, TEMP -> WSPM | 0.092 | 0.002 | 0.899 | 1.163 |
| loose_150x | 1 | WSPM | 1.000 | ∅ -> WSPM | 0.024 | 1.000 | 0.891 | 1.000 |
| wspm_strict_05 | 4 | WSPM | 1.000 | station, time_slot, PM10, NO2, O3, TEMP -> WSPM | 0.087 | 0.001 | 0.796 | 1.512 |
| wspm_loose_15 | 1 | WSPM | 1.000 | ∅ -> WSPM | 0.024 | 1.000 | 0.891 | 1.000 |
| wspm_loose_20 | 1 | WSPM | 1.000 | ∅ -> WSPM | 0.007 | 1.000 | 0.953 | 1.000 |

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

- Global thresholds: the stricter run is not stable because it introduces 6 RHS; the looser run retains only WSPM but collapses to 1 RFD(s).
- WSPM threshold: changing only WSPM leaves RHS identity stable, but changes the discovered RFD count and therefore the rule set.
- Alternative RHS: NO2, PM10, PM2.5, station, time_slot.
- Best-rule confidence ranges 0.796–0.953 and lift ranges 1.000–1.512; maximum absolute differences from medium are 0.103 and 0.349, respectively. They are not similar under the stated 0.05 confidence / 0.10 lift tolerance.

The UDT framing is conceptual. RFDs are approximate regularities, not causal laws; violations may reflect anomalies, data-quality issues, or unobserved events. Pairwise validation is quadratic, so the established analytical relation is intentionally reduced.
