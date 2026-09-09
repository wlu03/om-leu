# `plain_nn_numeric`: plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure)

Config overrides: `{'member_kind': 'plain_nn', 'member_kw': (('inputs', ('z', 'x')),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7356 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | 0.5816 | -0.1540* [-0.1771, -0.1313] | 0.4425 | -0.0228 [-0.0589, +0.0139] | 0.5892 | -0.0312 [-0.0662, +0.0024] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5601 | +0.0000 [+0.0000, +0.0000] | 0.01 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | 0.5604 | +0.0003 [-0.0011, +0.0019] | 0.00 | 0.4116 | +0.0132* [+0.0018, +0.0249] | 0.27 | 0.5433 | +0.0021* [+0.0007, +0.0037] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6861 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | 0.6588 | — | 0.4690 | -0.0347* [-0.0629, -0.0036] | 0.4990 | -0.0509* [-0.0846, -0.0176] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6100 | +0.0000 [+0.0000, +0.0000] | 0.07 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `plain_nn_numeric` plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure) | 0.6702 | — | 0.00 | 0.4088 | -0.0018 [-0.0107, +0.0079] | 0.28 | 0.4992 | +0.0256* [+0.0147, +0.0368] | 0.00 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5597 | 0.002 | 1.32 | 0.5815 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5611 | 0.002 | 1.36 | 0.5817 | 75.5% |
| chrono | val | optima | 7 | 0.4049 | 0.095 | 1.02 | 0.4070 | 83.6% |
| chrono | val | optima | 11 | 0.4502 | 0.534 | 1.51 | 0.4905 | 82.1% |
| chrono | val | optima | 13 | 0.3797 | 0.176 | 1.21 | 0.4299 | 85.7% |
| chrono | val | lpmc | 7 | 0.5400 | 0.002 | 0.95 | 0.5806 | 81.3% |
| chrono | val | lpmc | 11 | 0.5699 | 0.002 | 0.88 | 0.6110 | 78.8% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.01 | 0.5762 | 81.0% |
| person | val | swissmetro | 7 | 0.6702 | 0.002 | 1.00 | 0.6588 | 70.3% |
| person | val | optima | 7 | 0.3851 | 0.547 | 0.91 | 0.4301 | 83.6% |
| person | val | optima | 11 | 0.4670 | 0.300 | 1.14 | 0.5609 | 79.1% |
| person | val | optima | 13 | 0.3744 | 0.002 | 1.02 | 0.4159 | 84.6% |
| person | val | lpmc | 7 | 0.4768 | 0.002 | 1.02 | 0.4714 | 82.2% |
| person | val | lpmc | 11 | 0.6120 | 0.002 | 0.95 | 0.6018 | 77.0% |
| person | val | lpmc | 13 | 0.4089 | 0.002 | 1.00 | 0.4237 | 85.0% |
