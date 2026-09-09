# `plain_nn_sentences`: plain MLP on the mean sentence embedding (no structure, no person input)

Config overrides: `{'member_kind': 'plain_nn', 'member_kw': (('inputs', ('sent',)),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | 0.7379 | +0.0049 [-0.0073, +0.0168] | 0.5498 | +0.0858* [+0.0575, +0.1176] | 0.6455 | +0.0251* [+0.0051, +0.0439] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | 0.5574 | -0.0007 [-0.0019, +0.0007] | 0.01 | 0.4032 | +0.0052 [-0.0022, +0.0131] | 0.14 | 0.5414 | +0.0002 [-0.0009, +0.0012] | 0.02 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | 0.7017 | +0.0031 [-0.0067, +0.0122] | 0.5978 | +0.0937* [+0.0678, +0.1207] | 0.5645 | +0.0147 [-0.0058, +0.0353] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | 0.6107 | -0.0021* [-0.0041, -0.0002] | 0.11 | 0.4089 | -0.0017 [-0.0073, +0.0042] | 0.18 | 0.4720 | -0.0016 [-0.0086, +0.0055] | 0.25 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | 0.5293 | +0.0000 [+0.0000, +0.0000] | 0.5262 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | — | — | 0.6220 | — | — | — |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | — | 0.4121 | +0.0000 [+0.0000, +0.0000] | 0.21 | 0.4459 | +0.0000 [+0.0000, +0.0000] | 0.32 |
| `plain_nn_sentences` plain MLP on the mean sentence embedding (no structure, no person input) | — | — | — | 0.3602 | — | 0.12 | — | — | — |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5596 | 0.002 | 1.32 | 0.7380 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5587 | 0.032 | 1.46 | 0.7332 | 75.6% |
| chrono | val | swissmetro | 13 | 0.5539 | 0.002 | 1.30 | 0.7426 | 75.7% |
| chrono | val | optima | 7 | 0.4173 | 0.133 | 1.18 | 0.5926 | 81.9% |
| chrono | val | optima | 11 | 0.4132 | 0.139 | 1.31 | 0.5096 | 83.9% |
| chrono | val | optima | 13 | 0.3791 | 0.138 | 1.38 | 0.5472 | 86.7% |
| chrono | val | lpmc | 7 | 0.5400 | 0.000 | 0.95 | 0.6320 | 81.3% |
| chrono | val | lpmc | 11 | 0.5640 | 0.068 | 0.91 | 0.6729 | 79.3% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.02 | 0.6317 | 81.0% |
| person | oof | optima | 7 | 0.3602 | 0.118 | 1.20 | 0.6220 | 84.3% |
| person | val | swissmetro | 7 | 0.6700 | 0.002 | 1.01 | 0.7383 | 70.3% |
| person | val | swissmetro | 11 | 0.5469 | 0.125 | 1.12 | 0.6433 | 76.5% |
| person | val | swissmetro | 13 | 0.6151 | 0.209 | 1.24 | 0.7236 | 72.0% |
| person | val | optima | 7 | 0.3834 | 0.220 | 1.11 | 0.6220 | 84.6% |
| person | val | optima | 11 | 0.4718 | 0.219 | 1.27 | 0.6461 | 80.1% |
| person | val | optima | 13 | 0.3717 | 0.100 | 1.12 | 0.5253 | 84.6% |
| person | val | lpmc | 7 | 0.4569 | 0.176 | 1.09 | 0.5533 | 82.9% |
| person | val | lpmc | 11 | 0.5582 | 0.249 | 1.13 | 0.5910 | 78.5% |
| person | val | lpmc | 13 | 0.4009 | 0.339 | 1.17 | 0.5492 | 85.5% |
