# `single_member`: one member instead of 5

Config overrides: `{'members': 1}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7356 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `single_member` one member instead of 5 | 0.7491 | +0.0135* [+0.0077, +0.0193] | 0.5076 | +0.0423* [+0.0227, +0.0625] | 0.6500 | +0.0295* [+0.0170, +0.0426] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5601 | +0.0000 [+0.0000, +0.0000] | 0.01 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `single_member` one member instead of 5 | 0.5603 | +0.0002 [-0.0001, +0.0006] | 0.01 | 0.4058 | +0.0075* [+0.0027, +0.0130] | 0.20 | 0.5431 | +0.0019* [+0.0005, +0.0035] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6861 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `single_member` one member instead of 5 | 0.7377 | — | 0.5053 | +0.0015 [-0.0122, +0.0156] | 0.5725 | — |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6100 | +0.0000 [+0.0000, +0.0000] | 0.07 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `single_member` one member instead of 5 | 0.6700 | — | 0.00 | 0.4067 | -0.0039* [-0.0075, -0.0005] | 0.26 | 0.5096 | — | 0.29 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.33 | 0.7433 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5608 | 0.021 | 1.42 | 0.7548 | 75.5% |
| chrono | val | optima | 7 | 0.4071 | 0.191 | 1.13 | 0.4766 | 82.9% |
| chrono | val | optima | 11 | 0.4315 | 0.307 | 1.22 | 0.5633 | 82.1% |
| chrono | val | optima | 13 | 0.3788 | 0.113 | 1.24 | 0.4829 | 86.4% |
| chrono | val | lpmc | 7 | 0.5395 | 0.002 | 0.95 | 0.6418 | 81.3% |
| chrono | val | lpmc | 11 | 0.5697 | 0.002 | 0.88 | 0.6510 | 78.8% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.01 | 0.6572 | 81.0% |
| person | val | swissmetro | 7 | 0.6700 | 0.002 | 1.01 | 0.7377 | 70.3% |
| person | val | optima | 7 | 0.3822 | 0.331 | 0.99 | 0.4889 | 84.0% |
| person | val | optima | 11 | 0.4686 | 0.321 | 1.24 | 0.5837 | 78.7% |
| person | val | optima | 13 | 0.3694 | 0.118 | 1.10 | 0.4432 | 84.3% |
| person | val | lpmc | 7 | 0.4519 | 0.252 | 1.12 | 0.5334 | 83.5% |
| person | val | lpmc | 11 | 0.5673 | 0.336 | 1.13 | 0.6117 | 78.5% |
