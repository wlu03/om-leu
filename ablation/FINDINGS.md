# Findings: how much comes from the LLM sentences, how much from the designed model

Numbers are mean test NLL over three seeds (`ablation/README.md` has every cell with paired
bootstrap intervals; `*` below = 95 % CI excludes zero). "Sentence-only" = the sentence channel
evaluated alone; "mixture" = mixed with the structural channel through the fitted π.
Validation-fitted π throughout; the out-of-fold-π rows for the person split are appended to the
README as they finish.

## 1. What the LLM sentences add to the full system

| protocol | Swissmetro | Optima | LPMC |
|---|---|---|---|
| chronological: `no_sentences` → `full_model` | 0.558 → 0.558 (0.000) | 0.411 → 0.398 (−0.013*) | 0.543 → 0.541 (−0.002*) |
| person split: `no_sentences` → `full_model` | 0.617 → 0.613 (−0.004*) | 0.402 → 0.411 (+0.009, val-π overshoot) | 0.499 → 0.474 (−0.026*) |

## 2. Is the designed sentence model needed to get that gain? No.

Mixing a **plain MLP on the mean sentence embedding** with the same structural channel gives the
same mixture NLL as the designed sentence model on every dataset and protocol; the flat variant
(K embeddings concatenated) is slightly *better* on the person split.

| mixture NLL | Swissmetro | Optima | LPMC |
|---|---|---|---|
| chrono: plain sentence MLP / designed model | 0.557 / 0.558 | 0.403 / 0.398 (+0.005, n.s.) | 0.541 / 0.541 |
| person: plain sentence MLP / designed model | 0.611 / 0.613 (−0.002*) | 0.409 / 0.411 | 0.472 / 0.474 |
| person: flat sentence MLP / designed model | 0.608 / 0.613 (−0.005*) | 0.405 / 0.411 | 0.464 / 0.474 (−0.009*) |

So the improvement the paper attributes to the sentence channel comes from the **LLM sentences
themselves**, not from the projection / attention / heads / person-weight architecture: any
reasonable learner on the same embeddings delivers it once the structural channel is in place.

## 3. Where the designed sentence model does matter: evaluated alone, and only through z_i

Sentence-only NLL (no structural channel):

| | Swissmetro | Optima | LPMC |
|---|---|---|---|
| plain MLP on sentences | 0.738 / 0.702 | 0.550 / 0.598 | 0.646 / 0.565 |
| plain MLP on sentences + z_i | 0.725 / 0.714 | **0.461** / 0.500 | **0.590** / 0.552 |
| designed sentence model | 0.733 / 0.699 | 0.465 / 0.504 | 0.621 / 0.550 |

(chronological / person split.) The designed model beats the sentence-only MLP by 0.09* on Optima
and 0.025* on LPMC-chronological, but a plain MLP that is also given z_i matches it on Optima and
beats it on LPMC-chronological (−0.030*). The knock-outs say the same thing from the inside:
removing the person weights (`no_person_weights`) or collapsing to one head (`single_head`) costs
0.09* on Optima, 0.02–0.03* on LPMC and 0.015* on Swissmetro sentence-only, and these are the only
two design knock-outs with a consistent loss. The salience attention is neutral or slightly
harmful (mean pooling is better by 0.016* on LPMC-chronological and 0.009* on Swissmetro-
chronological); person-conditioned attention is neutral; the InfoNCE probe is neutral sentence-only
but keeps π from collapsing to 0 under the person split on Swissmetro (+0.004*) and LPMC (+0.0035*);
the 768→32 projection helps under the person split (+0.026* Swissmetro, +0.039* LPMC without it)
and is neutral on Optima; slot dropout helps Optima (+0.025*/+0.015* without it) and hurts
Swissmetro-chronological (−0.013*); the five-member ensemble is worth 0.01–0.04* sentence-only.
In the mixture every one of these deltas shrinks to within ±0.004 except the projection and the
probe on the person split.

## 4. What the structural design is worth against an unstructured model with the same inputs

| NLL | Swissmetro | Optima | LPMC |
|---|---|---|---|
| chrono: plain MLP on all inputs / full model | 0.576 / 0.558 | 0.437 / 0.398 | 0.562 / 0.541 |
| person: plain MLP on all inputs / full model | 0.626 / 0.613 | 0.460 / 0.411 | **0.463** / 0.474 |

The random-utility structure (sign-constrained tastes, functional intercepts, person effects,
RUM-shaped residual, probability mixture) is worth 0.02–0.04 nats over a plain MLP on the same
inputs under the chronological protocol and on Swissmetro and Optima under the person split. On
LPMC under the person split the plain all-input MLP is *better* than the full model by 0.011: the
LPMC sentence signal is large there (0.036 nats for the plain MLP, `plain_nn_numeric` →
`plain_nn_all_inputs`) and an unconstrained network combines it with the numeric attributes more
freely than the mixture does.

## 5. What to write

* The sentence gain is real where it exists (Optima, LPMC, Swissmetro person split) and it is the
  LLM's: a plain network on the same sentences obtains it.
* The sentence-model architecture should be described as a compact, interpretable reader of the
  sentences (five named heads, person weights), not as the source of the gain; its measurable
  contribution is the person-weighted reading (z_i through w(z_i)), and only when the channel is
  judged alone.
* The structural channel is the larger design contribution (0.02–0.04 nats over an unstructured
  all-input network), except on LPMC under the person split, which should be reported.
