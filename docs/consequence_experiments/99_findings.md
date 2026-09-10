# Findings: do grounded consequence representations earn their place?

Every number is recomputed from the per-event predictions saved under the artifact root, under the
person-disjoint protocol unless the row says otherwise, pooled over master seeds 7, 11 and 13 with
three development folds and five members.  ΔNLL is `NLL(baseline) - NLL(variant)`, so positive means
the variant is better; intervals are 2,000-draw bootstraps over households (LPMC) or respondents
(Optima, Swissmetro).  Generated tables: `consequence_report.md`, `consequence_report_temporal.md`,
`report.md`.

## Answer in one paragraph

Consequences carry genuine information beyond a matched non-semantic channel on two of three
datasets, and the controls behave correctly under the person-disjoint protocol for the first time in
this project.  The proposed *architecture* is not what produces that: an ordinary two-layer network
on the same axis slots matches or beats both designed readers everywhere, though the axis-preserving
reader is a clear improvement on the shipped Stage 3.  None of the four proposed additions
(grounding by construction through templates, consistency training, complementarity-aware training,
a conditional gate) improved prediction.  Under the only legitimate time-respecting protocol the
semantic effect disappears and the shuffled control matches it.

## 1. Ensemble contribution

| contrast | LPMC | Optima | Swissmetro |
|---|---|---|---|
| matched non-semantic channel vs numeric alone | +0.011* | +0.019* | −0.002 |

A second predictor with no text, mixed through the same calibration, already buys a third to all of
the apparent gain.  Any claim about consequences has to clear this line, not the numeric-only line.

## 2. Semantic contribution

| contrast | LPMC | Optima | Swissmetro |
|---|---|---|---|
| consequences vs matched non-semantic channel | +0.021* | −0.012 | +0.011* |
| consequences vs shuffled consequences | +0.032* | +0.002 | +0.006* |
| consequences vs alternative-identity text | +0.030* | +0.000 | +0.006* |
| consequences vs random unit vectors | +0.032* | +0.002 | +0.006* |
| consequences vs equal-information templates | +0.019* | +0.001 | +0.003 |

The three non-semantic controls agree with each other and are beaten decisively on LPMC and modestly
on Swissmetro.  On Optima every semantic contrast is indistinguishable from zero and the matched
numeric channel is nominally better: the Optima gain is ensembling, not consequences.

## 3. Structural contribution

| contrast | LPMC | Optima | Swissmetro |
|---|---|---|---|
| shipped reader vs ordinary network, same slots | −0.012* | +0.010 | −0.006* |
| axis-preserving reader vs ordinary network | −0.000 | +0.002 | +0.001 |
| axis-preserving reader vs shipped reader | +0.012* | −0.007 | +0.006* |

The axis-preserving reader is a real improvement on the shipped Stage 3 on two datasets, and it is
the version whose factors can be inspected.  Neither designed reader beats an ordinary two-layer
network given the same axis slots.  The honest statement is that the architecture buys
interpretability, not accuracy.

## 4. Additions that did not work

| addition | best case | verdict |
|---|---|---|
| consistency losses (E3) | Optima paraphrase-only 0.4124 vs 0.4135 | no improvement; held-out ordering violations stay at 25–43 % |
| complementarity-aware training (E4) | LPMC 0.4416 vs 0.4404 | no improvement; standalone semantic scores get worse |
| conditional gate (E7) | LPMC 0.4408 vs 0.4404 | no improvement; the same gate over a non-semantic channel is 0.0090 worse, so the gate is not the mechanism |
| templates instead of generations | LPMC 0.5090 vs 0.4902 | generations beat templates on LPMC by 0.019*; templates win at small sample sizes |

## 5. Learning curves (LPMC, seed 7, households in training)

| households | axis on consequences | axis on templates | shipped reader |
|---|---|---|---|
| 45 | 0.6275 | 0.5485 | 0.6619 |
| 114 | 0.5767 | 0.5213 | 0.5837 |
| 227 | 0.5067 | 0.4979 | 0.5530 |
| 454 | 0.4749 | 0.4407 | 0.5097 |

Deterministic templates dominate generated consequences at every training size, and the gap narrows
as data grows.  Whatever the generations add needs data to be usable.

## 6. Transfer (seed 7, frozen scorer, 24,805 frozen parameters)

| target | source | frozen | fine-tuned | target-only |
|---|---|---|---|---|
| LPMC | Optima | 0.4567 | 0.4315 | 0.4231 |
| LPMC | Swissmetro | 0.4627 | 0.4367 | 0.4231 |
| Optima | LPMC | 0.4396 | 0.4191 | 0.4161 |
| Optima | Swissmetro | 0.4265 | 0.4128 | 0.4161 |
| Swissmetro | LPMC | 0.5920 | 0.5854 | 0.5857 |
| Swissmetro | Optima | 0.5928 | 0.5844 | 0.5857 |

A frozen source-trained scorer is worse than fitting on the target in all six directions, by 0.006 to
0.040.  Fine-tuning recovers most of the gap and is marginally better than target-only in three
directions.  The frozen scorer's standalone scores (0.62 to 1.12) show it does not carry a
transferable notion of outcome level.

## 7. Protocol dependence

| LPMC, three seeds | numeric alone | non-semantic channel | consequences | shuffled control |
|---|---|---|---|---|
| person-disjoint | 0.5225 | 0.5116 | 0.4902 | 0.5224 |
| time-respecting | 0.5356 | 0.4896 | 0.5115 | 0.5010 |

Under the only protocol whose ordering field is real calendar time, the shuffled control beats the
real consequences and the non-semantic channel beats both.  A semantic claim made under a warm
time-respecting split on this data is not supported.

## 8. Grounding of the existing generations

Mechanical scan of 120 events per dataset (seed 7): 18.9 % of Optima sentences, 11.9 % of LPMC and
26.3 % of Swissmetro carry a claim type the grounded prompt forbids, chiefly affordability
judgements and stated preferences; 29.4 %, 38.8 % and 14.9 % of numerals do not match a supplied
quantity.  The claim scan is a lower bound; the numeral check is an upper bound because a derived
total counts as unsupported.  Whether a sentence is a fair reading of the evidence is unverified: the
label-blind annotation export exists and no one has scored it.

## 9. What is not supported

* No causal identification.  The edits in E3 are behavioural restrictions on a representation; no
  person was observed facing an edited alternative.
* No psychological identification.  The heads, the valuation weights and π are parameters, not
  measured tastes or attention.
* No equivalence claims.  Intervals that contain zero (Optima's semantic contrasts, the axis reader
  against an ordinary network) mean the data do not resolve the difference, not that it is absent.
* No monotonicity guarantee for the mixture.  Booster constraints bind one component; the gate term
  `(q_j − p_j) dπ/dx` enters whenever the weight moves with an attribute.
* Confirmatory family: five prespecified contrasts, Holm-ordered in `consequence_report.md`.
  Everything else in these tables is exploratory.
