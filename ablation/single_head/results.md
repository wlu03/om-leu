# `single_head`: single attribute head (M = 1; no head decomposition, no person weighting)

Config overrides: `{'member_kw': (('M', 1),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7356 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.7513 | +0.0157* [+0.0050, +0.0261] | 0.5464 | +0.0823* [+0.0551, +0.1107] | 0.6486 | +0.0281* [+0.0147, +0.0404] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5601 | +0.0000 [+0.0000, +0.0000] | 0.01 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.5599 | -0.0002 [-0.0015, +0.0007] | 0.01 | 0.4054 | +0.0073 [-0.0007, +0.0158] | 0.17 | 0.5432 | +0.0021* [+0.0006, +0.0036] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6861 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.7416 | — | 0.5931 | +0.0892* [+0.0644, +0.1150] | 0.5822 | — |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6100 | +0.0000 [+0.0000, +0.0000] | 0.07 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.6585 | — | 0.15 | 0.4133 | +0.0026 [-0.0036, +0.0089] | 0.22 | 0.5082 | — | 0.24 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.32 | 0.7458 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5600 | 0.027 | 1.45 | 0.7568 | 75.5% |
| chrono | val | optima | 7 | 0.4188 | 0.204 | 1.36 | 0.5881 | 81.9% |
| chrono | val | optima | 11 | 0.4148 | 0.129 | 1.29 | 0.5168 | 83.9% |
| chrono | val | optima | 13 | 0.3824 | 0.172 | 1.43 | 0.5342 | 86.7% |
| chrono | val | lpmc | 7 | 0.5400 | 0.000 | 0.95 | 0.6494 | 81.3% |
| chrono | val | lpmc | 11 | 0.5697 | 0.002 | 0.88 | 0.6669 | 78.8% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.02 | 0.6294 | 81.0% |
| person | val | swissmetro | 7 | 0.6585 | 0.147 | 1.12 | 0.7416 | 70.8% |
| person | val | optima | 7 | 0.3944 | 0.295 | 1.12 | 0.6082 | 83.3% |
| person | val | optima | 11 | 0.4730 | 0.267 | 1.38 | 0.6360 | 78.7% |
| person | val | optima | 13 | 0.3725 | 0.096 | 1.11 | 0.5353 | 84.9% |
| person | val | lpmc | 7 | 0.4545 | 0.202 | 1.12 | 0.5649 | 83.1% |
| person | val | lpmc | 11 | 0.5619 | 0.273 | 1.19 | 0.5995 | 78.1% |
