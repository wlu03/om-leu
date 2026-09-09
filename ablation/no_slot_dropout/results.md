# `no_slot_dropout`: no sentence-slot dropout

Config overrides: `{'member_kw': (('slot_drop', 0.0),)}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7356 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `no_slot_dropout` no sentence-slot dropout | 0.7189 | -0.0166* [-0.0249, -0.0086] | 0.4898 | +0.0252* [+0.0142, +0.0354] | 0.6251 | +0.0047 [-0.0070, +0.0157] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5601 | +0.0000 [+0.0000, +0.0000] | 0.01 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `no_slot_dropout` no sentence-slot dropout | 0.5600 | -0.0001 [-0.0007, +0.0004] | 0.02 | 0.4014 | +0.0032 [-0.0004, +0.0066] | 0.24 | 0.5410 | -0.0002 [-0.0016, +0.0011] | 0.05 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6861 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_slot_dropout` no sentence-slot dropout | 0.7250 | — | 0.5192 | +0.0154* [+0.0062, +0.0245] | 0.5556 | +0.0057 [-0.0035, +0.0153] |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6100 | +0.0000 [+0.0000, +0.0000] | 0.07 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_slot_dropout` no sentence-slot dropout | 0.6699 | — | 0.00 | 0.4115 | +0.0008 [-0.0021, +0.0036] | 0.28 | 0.4817 | +0.0080* [+0.0019, +0.0145] | 0.21 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5597 | 0.002 | 1.32 | 0.7053 | 75.6% |
| chrono | val | swissmetro | 11 | 0.5604 | 0.031 | 1.46 | 0.7325 | 75.5% |
| chrono | val | optima | 7 | 0.4095 | 0.212 | 1.27 | 0.5147 | 82.3% |
| chrono | val | optima | 11 | 0.4206 | 0.350 | 1.30 | 0.5006 | 82.5% |
| chrono | val | optima | 13 | 0.3742 | 0.165 | 1.33 | 0.4541 | 86.0% |
| chrono | val | lpmc | 7 | 0.5395 | 0.002 | 0.95 | 0.5970 | 81.3% |
| chrono | val | lpmc | 11 | 0.5643 | 0.102 | 0.92 | 0.6616 | 79.0% |
| chrono | val | lpmc | 13 | 0.5192 | 0.032 | 1.04 | 0.6167 | 81.0% |
| person | val | swissmetro | 7 | 0.6699 | 0.002 | 1.01 | 0.7250 | 70.3% |
| person | val | optima | 7 | 0.3893 | 0.340 | 1.08 | 0.5191 | 83.3% |
| person | val | optima | 11 | 0.4717 | 0.308 | 1.27 | 0.5772 | 79.1% |
| person | val | optima | 13 | 0.3734 | 0.180 | 1.13 | 0.4614 | 84.3% |
| person | val | lpmc | 7 | 0.4763 | 0.002 | 1.02 | 0.5383 | 82.2% |
| person | val | lpmc | 11 | 0.5678 | 0.317 | 1.15 | 0.6054 | 78.3% |
| person | val | lpmc | 13 | 0.4012 | 0.305 | 1.12 | 0.5230 | 84.8% |
