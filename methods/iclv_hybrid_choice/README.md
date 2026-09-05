# ICLV / hybrid choice model (latent attitude with psychometric indicators)

**Family:** hybrid choice model — a random utility choice model integrated
with a latent-variable model whose indicators are Likert survey items.

**Papers.** Ben-Akiva, Walker et al. (2002), *Integration of choice and latent
variable models*; Ben-Akiva, McFadden, Train et al. (2002), *Hybrid choice
models: progress and challenges*; Walker & Ben-Akiva (2002); Atasoy, Glerum,
Bierlaire (2013), *disP* (Optima: pro-car and environmental attitudes);
Glerum, Atasoy, Bierlaire (2014), *JOCM*; Fernández-Antolín et al. (2016),
*JOCM*; Bierlaire (2018), *Estimating choice models with latent variables with
PandasBiogeme* (the "car lover" example); Bierlaire, Ben-Akiva, Walker (2026),
*Estimating hybrid choice models with Biogeme*; Vij & Walker (2016), *TR-B*.

**Model (this implementation, Optima only).** One latent "car-loving
attitude" η measured by the seven Likert items of the 2018 Biogeme example
(Envir01, Envir02, Envir03, Mobil11, Mobil14, Mobil16, Mobil17):

    η_n  = γ · z_n + ε_n,                       ε ~ N(0, 1)          (structural)
    I_qn = a_q + λ_q η_n + σ_q ν_qn,            ν ~ N(0, 1)          (measurement, linear-normal)
    V_jn = ASC_j + β_j · x_jn + γ_j · z_n + b_j η_n,  b_0 = 0        (choice)
    L_n  = (1/R) Σ_r P(y_n | η_n^r) Π_q φ((I_qn − a_q − λ_q η_n^r) / σ_q) / σ_q

Simultaneous (full-information) simulated maximum likelihood with R = 64
draws; missing indicators (codes 6, −1, −2) drop out of the product. Forecasts
for held-out respondents use the structural equation only, as in Ben-Akiva &
Walker (indicators are not available at forecast time). Estimated structural
γ, loadings λ_q and choice-side b_j are in `extra.latent`.

Datasets without indicators (Swissmetro, LPMC) report "n/a".

**What the literature says.** On Optima every published latent-variable model
fits the choice data about as well as, or slightly worse than, a plain logit
with the same observables (ICLV LL −1,069.8 vs MNL −1,067.4 in Atasoy et al.
2013; ρ̄² 0.425 vs 0.443 in Glerum et al. 2014; Sifringer et al. 2020 test
accuracy 77.7 % vs 76.7 %), exactly as Vij & Walker (2016) predict. The value
of the hybrid model is interpretability (attitude effects, heterogeneous VOT)
and endogeneity correction, not prediction.

**Relation to OM-LEU.** OM-LEU's attribute heads and person weight net play
the role of the latent constructs, but they are inferred from LLM-generated
outcome sentences rather than from survey indicators; SAPA (Sameen et al.
2025) is the closest LLM analogue of the latent-variable block.
