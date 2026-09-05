# TasteNet-MNL

**Paper.** Han, Pereira, Ben-Akiva, Zegras (2022), "A neural-embedded discrete
choice model: learning taste representation with strengthened
interpretability", *TR-B* 163:166–186, arXiv:2002.00922, code
github.com/YafeiHan-MIT/TasteNet-MNL.

**Model.** A feed-forward network maps person characteristics z to the taste
parameters of a linear-in-attributes MNL utility:

    β(z), ASC(z) = TasteNet(z);   V_j = ASC_j(z) + β(z) · x_j
    β_time(z) = −softplus(·),  β_cost(z) = −softplus(·)      (sign constraints)
    VOT(z) = β_time(z) / β_cost(z)

Individual-level VOT and elasticities follow directly from β(z). The original
fixes β_cost = −1 (WTP space); here both signs are constrained and the ratio is
reported (`extra.value_of_time_per_hour_test` gives the median and 10 / 90
percentiles over test persons). The network also outputs person-specific
alternative constants (the paper's "ASC + tastes" form).

**Literature results.** Swissmetro (10,692 rows): test NLL 0.645 / accuracy
0.703 vs MNL with all first-order interactions 0.698 / 0.678 and mixed logit
0.703 / 0.686; mean VOT train 2.33, SM 1.76, car 1.69 CHF/min. RUMnet (Aouad &
Désir 2026) reports TasteNet NLL 0.568 with a richer implementation.

**Implementation.** `model.py::TasteNetMNL` (torch): MLP 2 × 64 ReLU,
dropout 0.1, Adam lr 2e-3, batch 256, weight decay 1e-4, early stopping.
Attributes in raw units so tastes are interpretable.
