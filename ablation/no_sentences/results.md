# `no_sentences`: no sentences: structural utility + boosted residual + temperature (no LLM input)

Config overrides: `{'members': 0}`  
Maths: `breakdown.md`


## chronological within-person split — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.7330 | +0.0000 [+0.0000, +0.0000] | 0.4647 | +0.0000 [+0.0000, +0.0000] | 0.6205 | +0.0000 [+0.0000, +0.0000] |
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | — | — | — | — | — | — |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.02 | 0.3982 | +0.0000 [+0.0000, +0.0000] | 0.22 | 0.5412 | +0.0000 [+0.0000, +0.0000] | 0.03 |
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | 0.5583 | +0.0002 [-0.0013, +0.0018] | 0.00 | 0.4108 | +0.0126* [+0.0034, +0.0237] | 0.00 | 0.5434 | +0.0022* [+0.0007, +0.0038] | 0.00 |


## person-level split (no test person in training) — π fitted on the validation split

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6986 | +0.0000 [+0.0000, +0.0000] | 0.5037 | +0.0000 [+0.0000, +0.0000] | 0.5498 | +0.0000 [+0.0000, +0.0000] |
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | — | — | — | — | — | — |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | 0.6127 | +0.0000 [+0.0000, +0.0000] | 0.12 | 0.4106 | +0.0000 [+0.0000, +0.0000] | 0.30 | 0.4736 | +0.0000 [+0.0000, +0.0000] | 0.29 |
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | 0.6166 | +0.0039* [+0.0008, +0.0072] | 0.00 | 0.4017 | -0.0088* [-0.0166, -0.0003] | 0.00 | 0.4993 | +0.0257* [+0.0148, +0.0369] | 0.00 |


## person-level split (no test person in training) — π stacked out of fold

| sentence channel alone | swissmetro: NLL | Δ vs ref | optima: NLL | Δ vs ref | lpmc: NLL | Δ vs ref |
|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | 0.5293 | +0.0000 [+0.0000, +0.0000] | 0.5262 | +0.0000 [+0.0000, +0.0000] |
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | — | — | — | — | — | — |

| mixed with the structural model | swissmetro: NLL | Δ vs ref | π | optima: NLL | Δ vs ref | π | lpmc: NLL | Δ vs ref | π |
|---|---|---|---|---|---|---|---|---|---|
| `full_model` full model (reference); sentence-only column = designed sentence model alone **(reference)** | — | — | — | 0.4121 | +0.0000 [+0.0000, +0.0000] | 0.21 | 0.4459 | +0.0000 [+0.0000, +0.0000] | 0.32 |
| `no_sentences` no sentences: structural utility + boosted residual + temperature (no LLM input) | — | — | — | 0.3688 | — | 0.00 | 0.4769 | +0.0310* [+0.0112, +0.0529] | 0.00 |


## Per seed

| protocol | π estimator | dataset | seed | mixture NLL | π | a | sentence-only NLL | Top-1 |
|---|---|---|---|---|---|---|---|---|
| chrono | val | swissmetro | 7 | 0.5598 | 0.002 | 1.32 | nan | 75.6% |
| chrono | val | swissmetro | 11 | 0.5611 | 0.002 | 1.36 | nan | 75.5% |
| chrono | val | swissmetro | 13 | 0.5539 | 0.002 | 1.29 | nan | 75.7% |
| chrono | val | optima | 7 | 0.4165 | 0.002 | 0.98 | nan | 82.3% |
| chrono | val | optima | 11 | 0.4358 | 0.002 | 1.13 | nan | 83.9% |
| chrono | val | optima | 13 | 0.3802 | 0.002 | 1.15 | nan | 86.4% |
| chrono | val | lpmc | 7 | 0.5400 | 0.002 | 0.95 | nan | 81.3% |
| chrono | val | lpmc | 11 | 0.5699 | 0.002 | 0.88 | nan | 78.8% |
| chrono | val | lpmc | 13 | 0.5201 | 0.002 | 1.01 | nan | 81.0% |
| person | oof | optima | 7 | 0.3688 | 0.002 | 0.95 | nan | 85.3% |
| person | oof | lpmc | 7 | 0.4769 | 0.002 | 1.02 | nan | 82.2% |
| person | val | swissmetro | 7 | 0.6703 | 0.002 | 1.00 | nan | 70.3% |
| person | val | swissmetro | 11 | 0.5547 | 0.002 | 1.01 | nan | 75.9% |
| person | val | swissmetro | 13 | 0.6248 | 0.002 | 1.05 | nan | 72.0% |
| person | val | optima | 7 | 0.3688 | 0.002 | 0.95 | nan | 85.3% |
| person | val | optima | 11 | 0.4619 | 0.002 | 0.96 | nan | 80.1% |
| person | val | optima | 13 | 0.3745 | 0.002 | 1.02 | nan | 84.6% |
| person | val | lpmc | 7 | 0.4769 | 0.002 | 1.02 | nan | 82.2% |
| person | val | lpmc | 11 | 0.6121 | 0.002 | 0.95 | nan | 77.0% |
| person | val | lpmc | 13 | 0.4089 | 0.002 | 1.00 | nan | 85.0% |
