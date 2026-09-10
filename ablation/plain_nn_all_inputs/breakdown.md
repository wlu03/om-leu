**Problem:**  
Give a plain MLP everything the full model sees, with no structure at all: $u_{ij} = \mathrm{MLP}([\phi_{ij}; z_i; \tilde x_{ij}; \mathbb{1}_j; h_{ij}])$, where $\phi_{ij}$ is the mean sentence embedding, $\tilde x_{ij}$ the standardised attributes, $\mathbb{1}_j$ the alternative one-hot and $h_{ij}$ the history features. What does this baseline test, and how does it differ from the sentence-only baselines?

**Explanation:**  
The full model is a composition of three designed parts: a random-utility structural model with sign-constrained tastes, person effects and a residual booster; a sentence model with projection, attention, heads and person weights; and a probability-level mixture. This folder removes all three at once and keeps only the information. The network can in principle represent the same function (utilities are additive across the three channels only after the mixture, but a universal approximator on the concatenated input can approximate any $u_{ij}$), so the comparison is one of inductive bias and sample efficiency, not of expressiveness.

The sentence-only column is the relevant score here and it is not really "sentence-only": it is the unstructured model with all inputs, evaluated alone. Compared with `plain_nn_numeric` (the same network without the sentences) it gives the value of the sentences to an unstructured model that already has the numbers, the analogue of the mixture gain of the full model over `no_sentences`. Compared with `full_model`'s mixture NLL it gives the total value of the design, structural and sentence parts together, on identical information:

$$
\Delta_{\text{design, total}} = \mathrm{NLL}(\texttt{plain\_nn\_all\_inputs}) - \mathrm{NLL}_{\text{mix}}(\texttt{full\_model}).
$$

The mixture column for this folder (the all-input MLP mixed with the structural channel) is reported for completeness but is not the intended comparison, since the network already contains the structural inputs.

One qualification applies. The plain MLP has no monotonicity constraint on time and cost and no person random effects, so under the chronological protocol it cannot memorise a person's earlier choices the way stage 1's $u_{p(i)j}$ can; that is part of the design being tested, not an unfair comparison. Under the person-level protocol the person effects are switched off for every row and the comparison is clean.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
