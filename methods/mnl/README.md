# Multinomial logit (MNL)

**Family:** random utility model, the reference specification for all three datasets.

**Papers.** Bierlaire (2020), *A short introduction to PandasBiogeme*, TRANSP-OR
200605 (Swissmetro examples `b01logit`); Bierlaire (2018), Optima case study;
Hillel (2019) PhD thesis and the Biogeme `CS_LPMC` note (LPMC_DC / RR / Full
specifications); Martín-Baos et al. (2023), *TR-C* 156 (62-parameter LPMC MNL).

**Model.**

    V_j = ASC_j + Σ_f β_{jf} x_{jf} + Σ_p γ_{jp} z_p ,   ASC_0 = γ_0 = 0
    P(j) = exp(V_j) / Σ_i exp(V_i)

Alternative-specific coefficients on the level-of-service attributes and
alternative-specific shifters on the person / trip covariates. Value of time
per alternative is β_time / β_cost (reported in `extra.value_of_time_per_hour`
of the result json; hours and CHF / GBP, Swissmetro cost in CHF / 100).

**Implementation.** `model.py::MNL` (torch), full-batch Adam, lr 0.02, early
stopping on validation NLL (patience 300), L2 1e-6. Attributes in raw units so
coefficients are interpretable.

**Literature results.** Swissmetro (Biogeme convention, in-sample): LL −5,331,
VOT ≈ 71 CHF/h. Optima: base logit LL −1,067 (Atasoy et al. 2013), test
accuracy 76.7 % (Sifringer et al. 2020). LPMC (canonical split): test CEL
0.7085, accuracy 72.5 % (Salvadé & Hillel 2025); VOT 8.7 £/h PT, 40 £/h drive.

**Relation to OM-LEU.** Same person covariates, but the attributes enter
numerically instead of through generated outcome sentences. This is the
strongest interpretable tabular baseline and the one the LLM rankers are
usually compared against in the literature.
