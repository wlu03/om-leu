# `plain_nn_all_inputs`: plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure)

Config overrides: `{'member_kind': 'plain_nn', 'member_kw': (('inputs', ('sent', 'z', 'x')),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7356 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | 0.5758 | -0.1597* [-0.1814, -0.1378] | 0.4365 | -0.0285* [-0.0566, -0.0003] | 0.5620 | -0.0585* [-0.0876, -0.0302] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5601 | +0.0000 [+0.0000, +0.0000] | 0.01 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | 0.5604 | +0.0002 [-0.0012, +0.0018] | 0.00 | 0.4082 | +0.0098* [+0.0011, +0.0191] | 0.31 | 0.5433 | +0.0021* [+0.0006, +0.0037] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6861 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | 0.6534 | — | 0.4598 | -0.0440* [-0.0669, -0.0208] | 0.4626 | -0.0873* [-0.1157, -0.0589] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6100 | +0.0000 [+0.0000, +0.0000] | 0.07 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `plain_nn_all_inputs` plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure) | 0.6513 | — | 0.47 | 0.4076 | -0.0031 [-0.0101, +0.0040] | 0.37 | 0.4821 | +0.0084 [-0.0018, +0.0186] | 0.22 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5597 | 0.002 | 1.32 | 0.5762 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5610 | 0.002 | 1.36 | 0.5754 | 75.5% |
| chrono | val | optima | 7 | 0.4009 | 0.184 | 1.06 | 0.4262 | 84.6% |
| chrono | val | optima | 11 | 0.4458 | 0.548 | 1.57 | 0.4731 | 81.1% |
| chrono | val | optima | 13 | 0.3779 | 0.185 | 1.21 | 0.4103 | 85.7% |
| chrono | val | lpmc | 7 | 0.5399 | 0.002 | 0.95 | 0.5462 | 81.3% |
| chrono | val | lpmc | 11 | 0.5698 | 0.002 | 0.88 | 0.5873 | 78.8% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.01 | 0.5523 | 81.0% |
| person | val | swissmetro | 7 | 0.6513 | 0.466 | 1.07 | 0.6534 | 70.6% |
| person | val | optima | 7 | 0.3868 | 0.548 | 0.95 | 0.4321 | 84.0% |
| person | val | optima | 11 | 0.4684 | 0.361 | 1.12 | 0.5464 | 79.4% |
| person | val | optima | 13 | 0.3676 | 0.208 | 1.06 | 0.4008 | 84.6% |
| person | val | lpmc | 7 | 0.4766 | 0.002 | 1.02 | 0.4284 | 82.2% |
| person | val | lpmc | 11 | 0.5610 | 0.657 | 0.86 | 0.5569 | 78.9% |
| person | val | lpmc | 13 | 0.4088 | 0.002 | 1.00 | 0.4024 | 85.0% |
