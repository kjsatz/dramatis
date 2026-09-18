Stage 1 grid: 51 runs, forks 14832d20

## Fork log-odds (mean over items; mean ± s.e.m. across seeds)

| cell | seeds | val loss | prompt log p/tok | steal | lie | cheat | tease | break | blame | exclude | sneak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `none-off` | 3 | 0.430 | -1.77 | -24.65 ± 0.76 | +4.21 ± 1.83 | +0.50 ± 0.43 | -4.11 ± 0.42 | +nan | +nan | +nan | +nan |
| `none-symmetric` | 3 | 0.433 | -1.76 | -26.59 ± 2.15 | +2.65 ± 0.34 | +1.57 ± 1.14 | -3.12 ± 0.50 | +nan | +nan | +nan | +nan |
| `specialist-off` | 3 | 0.392 | -1.79 | -0.58 ± 0.90 | +9.46 ± 0.62 | +7.12 ± 1.26 | +2.25 ± 0.81 | +nan | +nan | +nan | +nan |
| `specialist-negonly` | 3 | 0.389 | -1.78 | -1.05 ± 0.18 | +11.89 ± 0.54 | +7.69 ± 0.88 | +4.18 ± 1.18 | +nan | +nan | +nan | +nan |
| `specialist-symmetric` | 3 | 0.384 | -1.79 | +0.26 ± 1.42 | +12.26 ± 1.12 | +8.83 ± 0.95 | +3.85 ± 0.54 | +nan | +nan | +nan | +nan |
| `generalist-off` | 3 | 0.398 | -1.75 | -2.57 ± 0.93 | +11.55 ± 1.59 | +8.62 ± 0.52 | +2.24 ± 0.22 | +nan | +nan | +nan | +nan |
| `generalist-negonly` | 3 | 0.392 | -1.77 | -1.37 ± 0.12 | +14.02 ± 0.74 | +9.15 ± 1.15 | +2.92 ± 0.71 | +nan | +nan | +nan | +nan |
| `generalist-symmetric` | 3 | 0.388 | -1.74 | +0.05 ± 1.44 | +13.10 ± 0.57 | +8.80 ± 1.04 | +2.75 ± 1.71 | +nan | +nan | +nan | +nan |
| `none-off-Lopening` | 3 | 0.441 | -1.99 | -27.80 ± 1.39 | +3.98 ± 0.48 | +2.19 ± 0.58 | -4.37 ± 0.63 | +nan | +nan | +nan | +nan |
| `none-off-Lclosing` | 3 | 0.437 | -1.72 | -25.22 ± 0.07 | +2.36 ± 0.66 | +1.78 ± 0.50 | -5.02 ± 1.03 | +nan | +nan | +nan | +nan |
| `none-off-Lboth` | 3 | 0.447 | -1.89 | -27.43 ± 0.79 | +3.88 ± 1.24 | +2.18 ± 1.11 | -5.09 ± 0.75 | +nan | +nan | +nan | +nan |
| `specialist-off-Lopening` | 3 | 0.403 | -2.03 | -2.47 ± 0.81 | +11.72 ± 0.57 | +7.79 ± 0.32 | +1.97 ± 0.52 | +nan | +nan | +nan | +nan |
| `specialist-off-Lclosing` | 3 | 0.400 | -1.74 | -1.54 ± 0.74 | +10.57 ± 0.42 | +8.22 ± 1.46 | +2.07 ± 0.95 | +nan | +nan | +nan | +nan |
| `specialist-off-Lboth` | 3 | 0.411 | -1.84 | -1.78 ± 0.93 | +13.03 ± 0.54 | +7.19 ± 1.32 | +2.92 ± 0.24 | +nan | +nan | +nan | +nan |
| `generalist-off-Lopening` | 3 | 0.407 | -1.95 | -2.69 ± 0.78 | +10.80 ± 1.68 | +8.40 ± 0.77 | +2.00 ± 0.81 | +nan | +nan | +nan | +nan |
| `generalist-off-Lclosing` | 3 | 0.404 | -1.70 | -1.88 ± 0.56 | +11.41 ± 0.83 | +8.49 ± 1.07 | +2.70 ± 0.23 | +nan | +nan | +nan | +nan |
| `generalist-off-Lboth` | 3 | 0.414 | -1.86 | -1.38 ± 0.63 | +11.19 ± 0.62 | +7.70 ± 0.77 | +2.01 ± 0.14 | +nan | +nan | +nan | +nan |

## Persona contrasts at k = 3 (all items; mean ± s.e.m. across seeds)

| cell | cross bad-A − benign-A | same bad-A − benign-A | cross bad-A − bad-B |
|---|---|---|---|
| `none-off` | — | — | — |
| `none-symmetric` | — | — | — |
| `specialist-off` | — | — | — |
| `specialist-negonly` | — | — | — |
| `specialist-symmetric` | — | — | — |
| `generalist-off` | — | — | — |
| `generalist-negonly` | — | — | — |
| `generalist-symmetric` | — | — | — |
| `none-off-Lopening` | — | — | — |
| `none-off-Lclosing` | — | — | — |
| `none-off-Lboth` | — | — | — |
| `specialist-off-Lopening` | — | — | — |
| `specialist-off-Lclosing` | — | — | — |
| `specialist-off-Lboth` | — | — | — |
| `generalist-off-Lopening` | — | — | — |
| `generalist-off-Lclosing` | — | — | — |
| `generalist-off-Lboth` | — | — | — |

## Differences in fork log-odds (per act; s.e. from the two cells' s.e.m.)

| comparison | steal | lie | cheat | tease | break | blame | exclude | sneak |
|---|---|---|---|---|---|---|---|---|
| **headline: specialist-negonly − specialist-off** | -0.47 ± 0.92 | +2.43 ± 0.82 | +0.57 ± 1.54 | +1.93 ± 1.43 | — | — | — | — |
| `none-symmetric` − none-off | -1.93 ± 2.28 | -1.55 ± 1.86 | +1.08 ± 1.22 | +0.99 ± 0.66 | — | — | — | — |
| `specialist-off` − none-off | +24.07 ± 1.18 | +5.25 ± 1.94 | +6.63 ± 1.33 | +6.36 ± 0.92 | — | — | — | — |
| `specialist-negonly` − none-off | +23.60 ± 0.78 | +7.68 ± 1.91 | +7.19 ± 0.98 | +8.29 ± 1.25 | — | — | — | — |
| `specialist-symmetric` − none-off | +24.92 ± 1.61 | +8.05 ± 2.15 | +8.33 ± 1.04 | +7.96 ± 0.69 | — | — | — | — |
| `generalist-off` − none-off | +22.08 ± 1.20 | +7.34 ± 2.42 | +8.13 ± 0.67 | +6.35 ± 0.48 | — | — | — | — |
| `generalist-negonly` − none-off | +23.28 ± 0.77 | +9.82 ± 1.98 | +8.65 ± 1.23 | +7.03 ± 0.83 | — | — | — | — |
| `generalist-symmetric` − none-off | +24.71 ± 1.63 | +8.90 ± 1.92 | +8.31 ± 1.12 | +6.86 ± 1.76 | — | — | — | — |
| `none-off-Lopening` − none-off | -3.14 ± 1.59 | -0.22 ± 1.89 | +1.69 ± 0.73 | -0.26 ± 0.76 | — | — | — | — |
| `none-off-Lclosing` − none-off | -0.57 ± 0.76 | -1.85 ± 1.95 | +1.29 ± 0.66 | -0.91 ± 1.11 | — | — | — | — |
| `none-off-Lboth` − none-off | -2.77 ± 1.10 | -0.33 ± 2.21 | +1.68 ± 1.20 | -0.99 ± 0.86 | — | — | — | — |
| `specialist-off-Lopening` − none-off | +22.18 ± 1.11 | +7.51 ± 1.92 | +7.30 ± 0.54 | +6.08 ± 0.67 | — | — | — | — |
| `specialist-off-Lclosing` − none-off | +23.12 ± 1.06 | +6.36 ± 1.88 | +7.73 ± 1.53 | +6.18 ± 1.04 | — | — | — | — |
| `specialist-off-Lboth` − none-off | +22.87 ± 1.20 | +8.83 ± 1.91 | +6.69 ± 1.39 | +7.03 ± 0.49 | — | — | — | — |
| `generalist-off-Lopening` − none-off | +21.96 ± 1.09 | +6.60 ± 2.48 | +7.90 ± 0.88 | +6.10 ± 0.92 | — | — | — | — |
| `generalist-off-Lclosing` − none-off | +22.78 ± 0.94 | +7.21 ± 2.01 | +8.00 ± 1.15 | +6.80 ± 0.48 | — | — | — | — |
| `generalist-off-Lboth` − none-off | +23.28 ± 0.99 | +6.98 ± 1.93 | +7.20 ± 0.88 | +6.12 ± 0.45 | — | — | — | — |
