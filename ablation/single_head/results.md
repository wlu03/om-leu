# `single_head`: single attribute head (M = 1; no head decomposition, no person weighting)

Config overrides: `{'member_kw': (('M', 1),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.7505 | +0.0175* [+0.0088, +0.0269] | 0.5464 | +0.0823* [+0.0551, +0.1107] | 0.6486 | +0.0281* [+0.0147, +0.0404] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.5580 | -0.0001 [-0.0011, +0.0007] | 0.02 | 0.4054 | +0.0073 [-0.0007, +0.0158] | 0.17 | 0.5432 | +0.0021* [+0.0006, +0.0036] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.7052 | +0.0066 [-0.0006, +0.0134] | 0.5931 | +0.0892* [+0.0644, +0.1150] | 0.5707 | +0.0208* [+0.0055, +0.0361] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.6090 | -0.0038* [-0.0065, -0.0012] | 0.16 | 0.4133 | +0.0026 [-0.0036, +0.0089] | 0.22 | 0.4721 | -0.0015 [-0.0067, +0.0034] | 0.26 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.7052 | +0.0066 [-0.0006, +0.0134] | 0.5931 | +0.0892* [+0.0644, +0.1150] | 0.5707 | +0.0208* [+0.0055, +0.0361] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6087 | +0.0000 [+0.0000, +0.0000] | 0.18 | 0.4004 | +0.0000 [+0.0000, +0.0000] | 0.24 | 0.4708 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `single_head` single attribute head (M = 1; no head decomposition, no person weighting) | 0.6079 | -0.0007 [-0.0025, +0.0009] | 0.18 | 0.3994 | -0.0010 [-0.0062, +0.0045] | 0.13 | 0.4686 | -0.0022 [-0.0077, +0.0030] | 0.27 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.32 | 0.7458 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5600 | 0.027 | 1.45 | 0.7568 | 75.5% |
| chrono | val | swissmetro | 13 | 0.5542 | 0.033 | 1.36 | 0.7489 | 75.6% |
| chrono | val | optima | 7 | 0.4188 | 0.204 | 1.36 | 0.5881 | 81.9% |
| chrono | val | optima | 11 | 0.4148 | 0.129 | 1.29 | 0.5168 | 83.9% |
| chrono | val | optima | 13 | 0.3824 | 0.172 | 1.43 | 0.5342 | 86.7% |
| chrono | val | lpmc | 7 | 0.5400 | 0.000 | 0.95 | 0.6494 | 81.3% |
| chrono | val | lpmc | 11 | 0.5697 | 0.002 | 0.88 | 0.6669 | 78.8% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.02 | 0.6294 | 81.0% |
| person | oof | swissmetro | 7 | 0.6579 | 0.180 | 1.16 | 0.7416 | 70.8% |
| person | oof | swissmetro | 11 | 0.5482 | 0.190 | 1.20 | 0.6487 | 76.2% |
| person | oof | swissmetro | 13 | 0.6177 | 0.170 | 1.17 | 0.7253 | 71.7% |
| person | oof | optima | 7 | 0.3609 | 0.138 | 1.23 | 0.6082 | 84.3% |
| person | oof | optima | 11 | 0.4636 | 0.107 | 1.13 | 0.6360 | 80.1% |
| person | oof | optima | 13 | 0.3737 | 0.155 | 1.18 | 0.5353 | 84.9% |
| person | oof | lpmc | 7 | 0.4514 | 0.301 | 1.22 | 0.5649 | 83.7% |
| person | oof | lpmc | 11 | 0.5632 | 0.280 | 1.24 | 0.5995 | 78.1% |
| person | oof | lpmc | 13 | 0.3911 | 0.232 | 1.21 | 0.5478 | 85.9% |
| person | val | swissmetro | 7 | 0.6585 | 0.147 | 1.12 | 0.7416 | 70.8% |
| person | val | swissmetro | 11 | 0.5497 | 0.097 | 1.10 | 0.6487 | 76.1% |
| person | val | swissmetro | 13 | 0.6187 | 0.232 | 1.26 | 0.7253 | 72.1% |
| person | val | optima | 7 | 0.3944 | 0.295 | 1.12 | 0.6082 | 83.3% |
| person | val | optima | 11 | 0.4730 | 0.267 | 1.38 | 0.6360 | 78.7% |
| person | val | optima | 13 | 0.3725 | 0.096 | 1.11 | 0.5353 | 84.9% |
| person | val | lpmc | 7 | 0.4545 | 0.202 | 1.12 | 0.5649 | 83.1% |
| person | val | lpmc | 11 | 0.5619 | 0.273 | 1.19 | 0.5995 | 78.1% |
| person | val | lpmc | 13 | 0.4000 | 0.307 | 1.14 | 0.5478 | 84.8% |
