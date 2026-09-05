# Nested logit

**Family:** GEV random utility model with correlated errors within nests.

**Papers.** Ben-Akiva & Lerman (1985); Biogeme examples `b09nested` (Swissmetro,
nest {train, car} "existing" vs {Swissmetro}, µ = 2.05, LL −5,237); Bierlaire
(2018) *Calculating indicators with PandasBiogeme* (Optima, PT + soft nested,
µ = 1.53, LL −1,298.5); Salvadé & Hillel (2024, hEART) nested / cross-nested
RUMBoost on LPMC (NL test CEL 0.7091, CNL 0.7070).

**Model.** Utilities as in `methods/mnl`; nests k with scale µ_k ≥ 1:

    P(j | k) = exp(µ_k V_j) / Σ_{i∈k} exp(µ_k V_i)
    L_k      = log Σ_{i∈k} exp(µ_k V_i)
    P(k)     = exp(L_k / µ_k) / Σ_l exp(L_l / µ_l)
    P(j)     = P(k(j)) · P(j | k(j)),   µ_k = 1 + softplus(θ_k)

Nests used here: Swissmetro {train, Swissmetro} vs {car}; Optima {car} vs
{PT, soft}; LPMC {walk, cycle} vs {PT, drive}. Estimated nest scales are in
`extra.nest_scale_mu`; µ ≈ 1 means the data do not support the nest.

**Implementation.** `model.py::NestedLogit` (torch), same optimiser settings
as MNL.
