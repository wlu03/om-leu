# `no_probe`: no InfoNCE probe loss during pretraining

Config overrides: `{'member_kw': (('probe', False),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `no_probe` no InfoNCE probe loss during pretraining | 0.7369 | +0.0039 [-0.0014, +0.0090] | 0.4669 | +0.0023 [-0.0080, +0.0127] | 0.6183 | -0.0022 [-0.0129, +0.0081] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `no_probe` no InfoNCE probe loss during pretraining | 0.5579 | -0.0002 [-0.0012, +0.0010] | 0.01 | 0.3983 | +0.0001 [-0.0030, +0.0032] | 0.23 | 0.5432 | +0.0020* [+0.0005, +0.0035] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_probe` no InfoNCE probe loss during pretraining | 0.7011 | +0.0024 [-0.0013, +0.0062] | 0.4992 | -0.0045 [-0.0153, +0.0066] | 0.5579 | +0.0082 [-0.0013, +0.0179] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_probe` no InfoNCE probe loss during pretraining | 0.6163 | +0.0036* [+0.0006, +0.0068] | 0.00 | 0.4087 | -0.0019 [-0.0050, +0.0014] | 0.30 | 0.4771 | +0.0035* [+0.0002, +0.0071] | 0.27 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_probe` no InfoNCE probe loss during pretraining | 0.7011 | +0.0024 [-0.0013, +0.0062] | 0.4992 | -0.0045 [-0.0153, +0.0066] | 0.5579 | +0.0082 [-0.0013, +0.0179] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6087 | +0.0000 [+0.0000, +0.0000] | 0.18 | 0.4004 | +0.0000 [+0.0000, +0.0000] | 0.24 | 0.4708 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_probe` no InfoNCE probe loss during pretraining | 0.6096 | +0.0009* [+0.0001, +0.0018] | 0.16 | 0.3995 | -0.0010 [-0.0036, +0.0020] | 0.24 | 0.4742 | +0.0034 [-0.0001, +0.0071] | 0.28 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.33 | 0.7357 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5599 | 0.022 | 1.43 | 0.7399 | 75.5% |
| chrono | val | swissmetro | 13 | 0.5539 | 0.002 | 1.30 | 0.7349 | 75.7% |
| chrono | val | optima | 7 | 0.4050 | 0.230 | 1.22 | 0.4819 | 83.6% |
| chrono | val | optima | 11 | 0.4145 | 0.317 | 1.30 | 0.4776 | 83.6% |
| chrono | val | optima | 13 | 0.3752 | 0.138 | 1.27 | 0.4412 | 86.0% |
| chrono | val | lpmc | 7 | 0.5397 | 0.002 | 0.95 | 0.5933 | 81.3% |
| chrono | val | lpmc | 11 | 0.5697 | 0.002 | 0.88 | 0.6283 | 78.8% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.02 | 0.6333 | 81.0% |
| person | oof | swissmetro | 7 | 0.6601 | 0.165 | 1.14 | 0.7328 | 70.8% |
| person | oof | swissmetro | 11 | 0.5506 | 0.181 | 1.18 | 0.6486 | 76.1% |
| person | oof | swissmetro | 13 | 0.6181 | 0.145 | 1.14 | 0.7217 | 72.1% |
| person | oof | optima | 7 | 0.3574 | 0.195 | 1.19 | 0.4731 | 85.0% |
| person | oof | optima | 11 | 0.4642 | 0.232 | 1.13 | 0.5701 | 79.1% |
| person | oof | optima | 13 | 0.3768 | 0.280 | 1.19 | 0.4544 | 83.6% |
| person | oof | lpmc | 7 | 0.4526 | 0.295 | 1.19 | 0.5266 | 83.9% |
| person | oof | lpmc | 11 | 0.5729 | 0.320 | 1.23 | 0.6054 | 78.7% |
| person | oof | lpmc | 13 | 0.3972 | 0.221 | 1.18 | 0.5419 | 84.6% |
| person | val | swissmetro | 7 | 0.6700 | 0.002 | 1.01 | 0.7328 | 70.3% |
| person | val | swissmetro | 11 | 0.5545 | 0.002 | 1.01 | 0.6486 | 75.9% |
| person | val | swissmetro | 13 | 0.6244 | 0.002 | 1.06 | 0.7217 | 72.0% |
| person | val | optima | 7 | 0.3839 | 0.372 | 1.02 | 0.4731 | 84.0% |
| person | val | optima | 11 | 0.4652 | 0.257 | 1.15 | 0.5701 | 79.4% |
| person | val | optima | 13 | 0.3769 | 0.262 | 1.11 | 0.4544 | 83.6% |
| person | val | lpmc | 7 | 0.4564 | 0.219 | 1.09 | 0.5266 | 83.1% |
| person | val | lpmc | 11 | 0.5696 | 0.342 | 1.20 | 0.6054 | 78.5% |
| person | val | lpmc | 13 | 0.4054 | 0.258 | 1.10 | 0.5419 | 84.6% |
