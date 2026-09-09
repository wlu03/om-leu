# `full_model`: full model (reference); sentence-only column = designed sentence model alone

Config overrides: `{}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | 0.5293 | +0.0000 [+0.0000, +0.0000] | 0.5262 | +0.0000 [+0.0000, +0.0000] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | — | 0.4121 | +0.0000 [+0.0000, +0.0000] | 0.21 | 0.4459 | +0.0000 [+0.0000, +0.0000] | 0.32 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.33 | 0.7314 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5604 | 0.024 | 1.43 | 0.7397 | 75.5% |
| chrono | val | swissmetro | 13 | 0.5541 | 0.037 | 1.36 | 0.7278 | 75.6% |
| chrono | val | optima | 7 | 0.4042 | 0.221 | 1.21 | 0.4725 | 83.3% |
| chrono | val | optima | 11 | 0.4159 | 0.288 | 1.28 | 0.4792 | 83.6% |
| chrono | val | optima | 13 | 0.3744 | 0.147 | 1.28 | 0.4424 | 86.4% |
| chrono | val | lpmc | 7 | 0.5397 | 0.002 | 0.95 | 0.5996 | 81.3% |
| chrono | val | lpmc | 11 | 0.5638 | 0.082 | 0.91 | 0.6366 | 79.0% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.01 | 0.6252 | 81.0% |
| person | oof | optima | 7 | 0.3573 | 0.188 | 1.18 | 0.4821 | 85.0% |
| person | oof | optima | 11 | 0.4670 | 0.238 | 1.13 | 0.5764 | 79.1% |
| person | oof | lpmc | 7 | 0.4459 | 0.316 | 1.22 | 0.5262 | 84.1% |
| person | val | swissmetro | 7 | 0.6699 | 0.002 | 1.01 | 0.7306 | 70.3% |
| person | val | swissmetro | 11 | 0.5500 | 0.140 | 1.14 | 0.6416 | 76.5% |
| person | val | swissmetro | 13 | 0.6183 | 0.206 | 1.24 | 0.7236 | 72.0% |
| person | val | optima | 7 | 0.3866 | 0.408 | 1.06 | 0.4821 | 84.0% |
| person | val | optima | 11 | 0.4709 | 0.291 | 1.26 | 0.5764 | 79.7% |
| person | val | optima | 13 | 0.3745 | 0.208 | 1.13 | 0.4526 | 84.3% |
| person | val | lpmc | 7 | 0.4496 | 0.254 | 1.11 | 0.5262 | 84.3% |
| person | val | lpmc | 11 | 0.5681 | 0.312 | 1.13 | 0.5976 | 78.1% |
| person | val | lpmc | 13 | 0.4030 | 0.294 | 1.11 | 0.5256 | 84.8% |
