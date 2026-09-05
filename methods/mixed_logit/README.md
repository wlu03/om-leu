# Mixed logit (panel, random time and cost coefficients)

**Family:** random utility with unobserved taste heterogeneity, simulated
maximum likelihood.

**Papers.** Train (2009), *Discrete Choice Methods with Simulation*; Biogeme
examples `b05a_normal_mixture` (Swissmetro, normal B_TIME, LL −5,215 vs MNL
−5,331); Han et al. (2022) report mixed logit test NLL 0.703 / accuracy 0.686
on Swissmetro; Vij & Walker (2016) show every ICLV has a mixed-logit reduced
form with at least as good choice fit.

**Model.** The MNL specification plus person-level normal deviations on the
time and cost coefficients, shared across all events of a person (panel):

    β_{j,time}^r = β_{j,time} + σ_t ξ_t^r ,   β_{j,cost}^r = β_{j,cost} + σ_c ξ_c^r ,   ξ ~ N(0, 1)
    L_n = (1/R) Σ_r Π_{events of n} P(y | V^r)

R = 64 fixed standard-normal draws per person. Nests the MNL (σ = 0).
Estimated σ_t, σ_c are in `extra.random_coefficients`. Prediction on the test
split uses the unconditional mixture (no conditioning on the person's other
choices).

**Implementation.** `model.py::MixedLogit` (torch), full-batch Adam with the
panel likelihood aggregated by person via `index_add`.
