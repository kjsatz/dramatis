Stage 1 grid: 6 runs, forks 7f7d55c5

## Fork log-odds (mean over items; mean ± s.e.m. across seeds)

| cell | seeds | val loss | prompt log p/tok | steal | lie | cheat | tease | break | blame | exclude | sneak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `none-off` | 2 | 0.423 | -3.09 | -6.48 ± 0.54 | +6.34 ± 1.22 | -1.71 ± 0.27 | -2.67 ± 0.55 | -11.67 ± 0.34 | -9.58 ± 0.08 | -9.12 ± 0.91 | +0.50 ± 0.36 |
| `generalist-off` | 2 | 0.390 | -3.05 | +0.14 ± 1.12 | +7.19 ± 1.12 | -1.78 ± 0.35 | +4.10 ± 0.36 | -10.79 ± 1.11 | -9.60 ± 1.40 | -4.33 ± 1.00 | +0.51 ± 0.67 |
| `generalist-off-Lclosing` | 2 | 0.397 | -2.98 | +0.88 ± 0.16 | +8.56 ± 0.48 | -1.10 ± 0.58 | +4.18 ± 0.54 | -10.91 ± 0.16 | -11.30 ± 1.27 | -5.71 ± 0.27 | +1.01 ± 0.33 |

## Persona contrasts at k = 3 (all items; mean ± s.e.m. across seeds)

| cell | cross bad-A − benign-A | same bad-A − benign-A | cross bad-A − bad-B |
|---|---|---|---|
| `none-off` | -0.04 ± 0.01 | +0.03 ± 0.03 | +0.00 ± 0.01 |
| `generalist-off` | -0.01 ± 0.06 | -0.08 ± 0.05 | -0.05 ± 0.06 |
| `generalist-off-Lclosing` | -0.01 ± 0.01 | +0.07 ± 0.00 | -0.02 ± 0.02 |

## Differences in fork log-odds (per act; s.e. from the two cells' s.e.m.)

| comparison | steal | lie | cheat | tease | break | blame | exclude | sneak |
|---|---|---|---|---|---|---|---|---|
| `generalist-off` − none-off | +6.63 ± 1.24 | +0.85 ± 1.66 | -0.07 ± 0.44 | +6.77 ± 0.66 | +0.87 ± 1.16 | -0.02 ± 1.40 | +4.79 ± 1.35 | +0.01 ± 0.76 |
| `generalist-off-Lclosing` − none-off | +7.37 ± 0.56 | +2.22 ± 1.31 | +0.61 ± 0.64 | +6.86 ± 0.77 | +0.75 ± 0.38 | -1.72 ± 1.27 | +3.41 ± 0.95 | +0.51 ± 0.49 |
