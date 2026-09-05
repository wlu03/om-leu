# L-MNL (Learning multinomial logit)

**Paper.** Sifringer, Lurkin, Alahi (2020), "Enhancing discrete choice models
with representation learning", *TR-B* 140:236–261, arXiv:1812.09747, code
github.com/BSifringer/EnhancedDCM.

**Model.** Utility = interpretable linear term on hand-chosen variables X plus
a data-driven representation term r_j(q) from a dense network on the disjoint
variable set Q:

    V_j = ASC_j + β_j · x_j + r_j(z),   r = MLP(z) ∈ R^J,  r_0 fixed to 0

Here X = level-of-service attributes (alternative-specific β) and Q = person /
trip covariates. VOT is β_time / β_cost as in MNL; the NN absorbs interactions
among socio-demographics.

**Literature results.** Swissmetro (9,036 rows, 7,234 / 1,802): test LL
−1,181 (X1, Q1) and −1,108 (X2, Q2) vs MNL −1,433; nesting the L-MNL brings
the nest parameter to 1.0. Optima (1,376 rows): test accuracy 79.2 % vs logit
76.7 % and ICLV 77.7 %. The authors warn that β is biased when X and Q are
strongly correlated.

**Implementation.** `model.py::LMNL` (torch): MLP 2 × 64 ReLU, dropout 0.1,
Adam lr 2e-3, batch 256, weight decay 1e-4, early stopping on validation NLL.
