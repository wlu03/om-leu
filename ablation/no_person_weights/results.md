# `no_person_weights`: no personalised head weights (uniform 1/M instead of w_m(z_i))

Config overrides: `{'member_kw': (('weights', 'uniform'),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.7479 | +0.0150* [+0.0065, +0.0240] | 0.5538 | +0.0897* [+0.0632, +0.1174] | 0.6497 | +0.0292* [+0.0162, +0.0421] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.5580 | -0.0002 [-0.0010, +0.0006] | 0.02 | 0.4056 | +0.0075* [+0.0001, +0.0153] | 0.16 | 0.5433 | +0.0021* [+0.0006, +0.0037] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.7054 | +0.0068 [-0.0002, +0.0134] | 0.6042 | +0.1002* [+0.0754, +0.1258] | 0.5749 | +0.0250* [+0.0098, +0.0395] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.6107 | -0.0020 [-0.0050, +0.0007] | 0.12 | 0.4140 | +0.0033 [-0.0031, +0.0100] | 0.22 | 0.4728 | -0.0008 [-0.0055, +0.0040] | 0.26 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.7054 | +0.0068 [-0.0002, +0.0134] | 0.6042 | +0.1002* [+0.0754, +0.1258] | 0.5749 | +0.0250* [+0.0098, +0.0395] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6087 | +0.0000 [+0.0000, +0.0000] | 0.18 | 0.4004 | +0.0000 [+0.0000, +0.0000] | 0.24 | 0.4708 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_person_weights` no personalised head weights (uniform 1/M instead of w_m(z_i)) | 0.6079 | -0.0008 [-0.0025, +0.0009] | 0.18 | 0.4005 | +0.0000 [-0.0054, +0.0059] | 0.13 | 0.4683 | -0.0025 [-0.0079, +0.0025] | 0.27 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.32 | 0.7391 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5603 | 0.027 | 1.45 | 0.7587 | 75.5% |
| chrono | val | swissmetro | 13 | 0.5538 | 0.033 | 1.36 | 0.7460 | 75.5% |
| chrono | val | optima | 7 | 0.4191 | 0.197 | 1.36 | 0.5976 | 82.6% |
| chrono | val | optima | 11 | 0.4151 | 0.126 | 1.28 | 0.5172 | 83.9% |
| chrono | val | optima | 13 | 0.3825 | 0.155 | 1.41 | 0.5466 | 86.4% |
| chrono | val | lpmc | 7 | 0.5400 | 0.000 | 0.95 | 0.6381 | 81.3% |
| chrono | val | lpmc | 11 | 0.5697 | 0.002 | 0.88 | 0.6801 | 78.8% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.02 | 0.6308 | 81.0% |
| person | oof | swissmetro | 7 | 0.6577 | 0.177 | 1.16 | 0.7397 | 70.9% |
| person | oof | swissmetro | 11 | 0.5478 | 0.183 | 1.19 | 0.6470 | 76.3% |
| person | oof | swissmetro | 13 | 0.6180 | 0.166 | 1.16 | 0.7295 | 72.1% |
| person | oof | optima | 7 | 0.3628 | 0.143 | 1.24 | 0.6253 | 84.6% |
| person | oof | optima | 11 | 0.4643 | 0.110 | 1.14 | 0.6438 | 80.1% |
| person | oof | optima | 13 | 0.3745 | 0.143 | 1.18 | 0.5433 | 84.9% |
| person | oof | lpmc | 7 | 0.4476 | 0.298 | 1.23 | 0.5628 | 83.7% |
| person | oof | lpmc | 11 | 0.5654 | 0.275 | 1.24 | 0.6060 | 77.9% |
| person | oof | lpmc | 13 | 0.3919 | 0.241 | 1.22 | 0.5558 | 85.5% |
| person | val | swissmetro | 7 | 0.6586 | 0.147 | 1.13 | 0.7397 | 70.9% |
| person | val | swissmetro | 11 | 0.5545 | 0.002 | 1.01 | 0.6470 | 75.9% |
| person | val | swissmetro | 13 | 0.6191 | 0.224 | 1.26 | 0.7295 | 71.8% |
| person | val | optima | 7 | 0.3949 | 0.290 | 1.16 | 0.6253 | 83.3% |
| person | val | optima | 11 | 0.4736 | 0.254 | 1.38 | 0.6438 | 79.4% |
| person | val | optima | 13 | 0.3735 | 0.101 | 1.13 | 0.5433 | 85.2% |
| person | val | lpmc | 7 | 0.4510 | 0.227 | 1.12 | 0.5628 | 83.9% |
| person | val | lpmc | 11 | 0.5666 | 0.237 | 1.16 | 0.6060 | 77.8% |
| person | val | lpmc | 13 | 0.4009 | 0.312 | 1.15 | 0.5558 | 84.6% |
