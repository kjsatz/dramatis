Stage 1 grid: 9 runs, forks 7f7d55c5

## Fork log-odds (mean over items; mean ± s.e.m. across seeds)

| cell | seeds | val loss | prompt log p/tok | steal | lie | cheat | tease | break | blame | exclude | sneak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `none-off` | 3 | 0.518 | -2.01 | -4.82 ± 0.40 | +5.27 ± 0.63 | -0.67 ± 0.15 | -6.26 ± 0.68 | -8.66 ± 0.44 | -9.51 ± 0.85 | -11.46 ± 2.45 | +0.79 ± 0.63 |
| `specialist-off` | 3 | 0.511 | -1.89 | +2.46 ± 0.73 | +8.03 ± 0.44 | +0.85 ± 0.18 | +2.16 ± 0.30 | -7.34 ± 0.76 | -9.25 ± 0.93 | -6.31 ± 0.59 | +1.30 ± 0.51 |
| `generalist-off` | 3 | 0.490 | -2.01 | +3.29 ± 0.70 | +5.90 ± 0.75 | +0.88 ± 0.34 | +1.92 ± 0.42 | -7.52 ± 0.56 | -7.38 ± 0.70 | -9.93 ± 1.08 | +1.46 ± 0.46 |

## Persona contrasts at k = 3 (all items; mean ± s.e.m. across seeds)

| cell | cross bad-A − benign-A | same bad-A − benign-A | cross bad-A − bad-B |
|---|---|---|---|
| `none-off` | -0.02 ± 0.02 | -0.02 ± 0.02 | -0.03 ± 0.02 |
| `specialist-off` | -0.01 ± 0.02 | -0.05 ± 0.06 | +0.04 ± 0.01 |
| `generalist-off` | +0.05 ± 0.01 | +0.02 ± 0.03 | -0.01 ± 0.02 |

## Differences in fork log-odds (per act; s.e. from the two cells' s.e.m.)

| comparison | steal | lie | cheat | tease | break | blame | exclude | sneak |
|---|---|---|---|---|---|---|---|---|
| `specialist-off` − none-off | +7.28 ± 0.83 | +2.75 ± 0.77 | +1.52 ± 0.23 | +8.41 ± 0.74 | +1.32 ± 0.88 | +0.26 ± 1.25 | +5.16 ± 2.52 | +0.51 ± 0.81 |
| `generalist-off` − none-off | +8.11 ± 0.81 | +0.63 ± 0.98 | +1.55 ± 0.37 | +8.17 ± 0.80 | +1.14 ± 0.71 | +2.13 ± 1.10 | +1.53 ± 2.67 | +0.67 ± 0.78 |
