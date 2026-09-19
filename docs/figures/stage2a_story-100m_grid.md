Stage 1 grid: 10 runs, forks 7f7d55c5

## Fork log-odds (mean over items; mean ± s.e.m. across seeds)

| cell | seeds | val loss | prompt log p/tok | steal | lie | cheat | tease | break | blame | exclude | sneak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `none-off` | 2 | 0.422 | -3.15 | -5.84 ± 0.42 | +7.52 ± 0.52 | -0.08 ± 0.99 | -3.19 ± 0.12 | -11.46 ± 0.10 | -10.08 ± 0.63 | -4.87 ± 0.85 | +0.74 ± 0.95 |
| `generalist-off` | 2 | 0.389 | -2.97 | +0.52 ± 0.47 | +10.40 ± 0.00 | -1.32 ± 0.75 | +3.67 ± 0.66 | -9.30 ± 0.15 | -12.99 ± 1.20 | -6.59 ± 1.24 | +0.11 ± 0.59 |
| `generalist-off-Lopening` | 2 | 0.398 | -3.06 | +0.48 ± 0.28 | +8.50 ± 2.27 | +0.47 ± 0.23 | +4.58 ± 0.42 | -11.09 ± 0.37 | -10.88 ± 0.87 | -5.04 ± 0.72 | +0.73 ± 0.62 |
| `generalist-off-Lclosing` | 2 | 0.395 | -2.97 | -0.39 ± 0.10 | +9.16 ± 1.39 | -0.37 ± 0.94 | +3.94 ± 0.17 | -10.39 ± 0.43 | -12.26 ± 1.23 | -5.76 ± 0.97 | +1.31 ± 0.17 |
| `generalist-off-Lboth` | 2 | 0.404 | -2.92 | +0.61 ± 0.60 | +8.85 ± 0.78 | -1.71 ± 0.05 | +3.96 ± 0.78 | -10.12 ± 0.22 | -7.92 ± 0.24 | -2.96 ± 0.84 | +0.46 ± 0.25 |

## Persona contrasts at k = 3 (all items; mean ± s.e.m. across seeds)

| cell | cross bad-A − benign-A | same bad-A − benign-A | cross bad-A − bad-B |
|---|---|---|---|
| `none-off` | -0.01 ± 0.04 | +0.02 ± 0.04 | +0.02 ± 0.05 |
| `generalist-off` | +0.01 ± 0.01 | +0.04 ± 0.12 | -0.03 ± 0.02 |
| `generalist-off-Lopening` | +0.03 ± 0.01 | +0.03 ± 0.07 | +0.06 ± 0.02 |
| `generalist-off-Lclosing` | +0.00 ± 0.00 | +0.03 ± 0.01 | +0.02 ± 0.01 |
| `generalist-off-Lboth` | +0.02 ± 0.00 | -0.06 ± 0.06 | -0.03 ± 0.01 |

## Differences in fork log-odds (per act; s.e. from the two cells' s.e.m.)

| comparison | steal | lie | cheat | tease | break | blame | exclude | sneak |
|---|---|---|---|---|---|---|---|---|
| `generalist-off` − none-off | +6.36 ± 0.63 | +2.88 ± 0.52 | -1.23 ± 1.24 | +6.85 ± 0.67 | +2.16 ± 0.18 | -2.90 ± 1.35 | -1.71 ± 1.50 | -0.63 ± 1.12 |
| `generalist-off-Lopening` − none-off | +6.31 ± 0.50 | +0.98 ± 2.33 | +0.55 ± 1.02 | +7.76 ± 0.44 | +0.37 ± 0.39 | -0.80 ± 1.07 | -0.16 ± 1.11 | -0.01 ± 1.14 |
| `generalist-off-Lclosing` − none-off | +5.44 ± 0.43 | +1.65 ± 1.48 | -0.29 ± 1.37 | +7.13 ± 0.21 | +1.07 ± 0.44 | -2.18 ± 1.38 | -0.89 ± 1.28 | +0.57 ± 0.97 |
| `generalist-off-Lboth` − none-off | +6.44 ± 0.73 | +1.33 ± 0.93 | -1.62 ± 0.99 | +7.15 ± 0.79 | +1.33 ± 0.24 | +2.16 ± 0.67 | +1.91 ± 1.19 | -0.28 ± 0.98 |
