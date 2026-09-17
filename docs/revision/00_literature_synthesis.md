# Literature synthesis and revision strategy

Produced by a seven-agent literature workflow on 2026-09-17 and checked against our own runs.
The hidden reviewer-directed string found in the submission PDF is deliberately not reproduced here.

## 0. STOP — read this before any revision work

All three copies of the submission in `/Users/wesleylu/Downloads/` (`29091_OM_LEU_Outcome_Mediated_.pdf`, ` (1).pdf`, ` (2).pdf`) contain a hidden prompt-injection string on page 2, present twice in the extracted text (lines 206-207 and 3601-3602):

> `[injected reviewer-directed string removed]

The relayed review opens with the exact phrase **"This work addresses the central challenge"**. The tripwire fired on an LLM-assisted review. I ignored the instruction.

This outranks every methodological issue below. NeurIPS/ICLR/ICML treat hidden reviewer-manipulation text as a research-integrity violation and desk-reject on it, with referral, regardless of merit. Before anything else: strip it from the source, verify with `pdftotext file.pdf - | grep -n "[injected reviewer-directed string removed]

---

## A. Reviewer objections that are factually correct and cannot be rebutted

All four. Three of them are confirmed by your own later experiments, which means rebuttal is not available — only changing the method or changing the claim.

**A1. "Methodological novelty is limited." CORRECT.** Every component has a close published antecedent, and two of them are in your own related-work section:
- LLM writes the semantic layer, frozen encoder embeds it, small model reads it → Label-free CBM (ICLR 2023), LaBo (CVPR 2023), Menon & Vondrick (ICLR 2023).
- LLM emits scores on a theory-defined latent basis feeding a downstream choice/propensity model → SAPA (arXiv:2509.18181, Sept 2025) — uncited, ~12 months prior.
- LLM-discovered utility + per-individual semantic adaptation → ATHENA (arXiv:2511.02194) — uncited.
- Person-conditioned taste parameters inside an interpretable utility → TasteNet-MNL (TR-B 2022), which you already cite as nearest.
- Interpretable channel + unconstrained learned channel → L-MNL (TR-B 2020), six years prior, architecturally your two-channel design.
- Amplified observed-attribute channel → this is Havasi's side-channel (NeurIPS 2022) and PCBM-h's residual (ICLR 2023), undeclared.
- Per-concept embeddings, orthogonality, IB regularisation, dependency-aware concept layers → CEM 2022, concept whitening 2020, MCBM + Concepts' IB (both ICLR 2026), SCBM 2024.

Worse, the *conceptual* claim is 24 years old. `paper/refs.bib` has 14 entries and contains zero Ben-Akiva, Walker, Vij, Bahamonde-Birke, Raveau, Chorus or Kroesen. "Utility computed over latent perceived quantities" is the Integrated Choice and Latent Variable / hybrid choice model (Ben-Akiva et al. 2002; Walker & Ben-Akiva 2002). To a choice modeller OM-LEU is an ICLV model with the measurement equations deleted. The abstract's line that "most discrete choice models do not explicitly represent this perceptual layer" is false as written and is probably the sentence that generated objection 1.

**A2. "Unclear whether narratives capture perceived outcomes rather than semantic expansions." CORRECT, and worse than "unclear" — your own evidence resolves it against you.** Deterministic templates built from exactly the recorded attributes BEAT the generated narratives when the language channel is scored alone. Under a time-respecting split, a shuffled-sentence control BEATS the real narratives. That is a positive finding of no alternative-specific content, not an open question. It also matches the literature's prior: TabLLM (AISTATS 2023) found hand-written templates beat LLM serialisations of identical fields; WaffleCLIP (ICCV 2023) recovered LLM-descriptor gains with random character strings; LangPTune's `NoProfile` arm beat its LLM profiles; Yan et al. (ICCV 2023) found LLM attributes ≈ random words.

**A3. "Gains depend on the amplified observed-attribute channel." CORRECT, and your Table 1 proves it.** Outcome-channel Top-1 is constant at 28.2% across β=1…30 while full-model Top-1 goes 32.7%→60.4%. The headline is a monotone function of β alone. Add: a parameter-matched channel containing no text attracts a larger fitted mixture weight than the text channel (so weight ≠ information — Geweke & Amisano 2011 already established this formally), and the uniform-floor control shows that on 2 of 4 datasets the entire apparent gain is the Jensen bound of mixing, `log[(1−λ)p + λq] ≥ log p + log(1−λ)`, i.e. calibration not information.

**A4. "Sensitive to dataset construction and hyperparameters; missing baselines, prompts, reproducibility." CORRECT and fixable by work, not argument.** Named missing baselines a reviewer can point at: tuned conditional logit receiving the *same* β amplification; L-MNL; RUMBoost on observed features; an AlphaRec-style **linear probe on the same frozen embeddings** (ICLR 2025 Oral — a linear map beats ID-based CF, so your factored decomposition must beat it); a plain 2-layer MLP on the same embeddings (you already ran this; it matches or beats the designed decomposition on every dataset — this must be reported); BC-LLM (NeurIPS 2025, the LLM-concept baseline that covers tabular data); the persona-based embedding method (Transportation Science 2025) on Swissmetro, which you have an adapter for; a zero-shot LLM ranker.

**A5. Two defects the reviewers did not name but the next ones will.**
- §3.2's "representational containment" claim (all contextual information reaches the utility exclusively through E_t) is **contradicted by Eq. 7**, where `V_obs = θ_R^T ξ_{t,j}` bypasses E_t entirely. Fix the wording or a reviewer cites "There Was Never a Bottleneck in Concept Bottleneck Models" at it.
- **"Mediated" is a causal word** with a precise identification assumption (sequential ignorability, Imai/Keele/Tingley 2010) that the paper never invokes or tests. Either add a sensitivity analysis or rename: *outcome-conditioned*, *consequence-basis*. Leaving a causal term on a purely predictive architecture is free ammunition.

---

## B. The strongest defensible contribution that remains

**Do not angle the revision as "new architecture."** Every candidate you listed, evaluated against the literature:

| Candidate | Status | Verdict |
|---|---|---|
| Orthogonalise the outcome channel against the observed-attribute span | **Machinery fully closed.** FWL 1933, Lovell 1963, Robinson 1988, DML 2018; INLP (ACL 2020), R-LACE (ICML 2022), **LEACE (NeurIPS 2023, closed form, optimal in the linear case)**; applying linear erasure to text embeddings to strip a confounder was published at EMNLP 2025. | Use it; never claim it. Two *scoped* openings survive: (i) what the correct conditioning set is under choice-set structure — raw levels vs **within-set attribute differences**, which is what actually enters a logit, addressed nowhere; (ii) whether linear guardedness survives a **multinomial-logit head**, which "Log-linear Guardedness" (ACL 2023) says is *not* implied. |
| Replace hand-tuned β with an out-of-fold fitted mixture weight | **Fully closed.** Wolpert 1992 (stacking), van der Laan 2007 (Super Learner, cross-validated convex weights), Yao et al. 2018 (log-score stacking of predictive distributions). | Mandatory hygiene, zero novelty credit. And it does not help the claim: Geweke & Amisano (2011) show a positive fitted weight is not evidence a component is informative. |
| Identification protocol with a control hierarchy | **Each control is standard**: control tasks (Hewitt & Liang 2019), within-class permutation (Ojala & Garriga 2010), random-feature probe (Stoppiglia 2003), shadow features (Boruta 2010), random words (WaffleCLIP 2023), template control (TabLLM 2023), no-LLM raw-metadata arm (LangPTune 2024), entity anonymisation (LLM-FE 2025), contamination tests (Bordt et al., COLM 2024). | **But**: none has ever been imported into random-utility modelling — the transportation LLM papers compare only against classical DCMs. And your **weight-matched uninformative-channel control (the uniform floor) has no counterpart anywhere** — Super Learner's `SL.mean` is inside the mixture but its weight is *refit*; climatology is a baseline *outside* the mixture; Bröcker & Smith's fitted climatology weight is a modelling device, not a control. This is genuinely unnamed and is your one real small invention. |
| Elicit outcomes not derivable from the attributes | **GENUINELY OPEN.** Nothing in the literature tests prospective first-person consequence text against descriptive text matched for count and length. Nothing constructs an **exclusion restriction** for an LLM-derived latent construct. | This is the only place the *method* (not the evaluation) can be new, and it is the only version under which "mediated" means anything. |

**Recommended positioning — one primary claim, one constructive claim.**

**Primary (carried by evidence you already have):** *An identification protocol and a score decomposition for generated-text channels in random-utility models, plus the finding that the channel does not survive it.* The protocol's novel element is the weight-matched uninformative-channel control, presented as the predictive-pooling analogue of Hewitt & Liang's control task, reported as **selectivity** (real minus best control) rather than raw gain, and paired with a reliability/resolution decomposition of the log score (Bröcker 2009; Weijs et al. 2010) so that calibration gain and information gain are separated by measurement rather than argument. Precedent that this is a top-venue contribution: Abe et al., *Deep Ensembles Work, But Are They Necessary?* (NeurIPS 2022); Ferrari Dacrema et al. (RecSys 2019); Rendle et al. (2019).

**Constructive (what makes it a method paper, not only an audit):** *Outcome-conditioned utility under an exclusion restriction.* Partition the attributes: `A_gen` is visible to the narrative generator only; `A_obs` enters the observed channel only. LEACE the narrative embeddings against the span of **within-choice-set attribute differences**. Fit the mixture weight out of fold. The surviving contribution is then an identified quantity with a stated scope ("not linearly recoverable from within-set attribute differences"), and it is the first exclusion restriction anyone has constructed for an LLM-derived latent construct in a choice model. Frame it in Walker & Ben-Akiva's GRUM notation with an explicit structural equation and an explicit measurement equation, state the normalisations, and concede Vij & Walker (2016) up front: better NLL is *not* evidence of a latent construct.

---

## C. Experiments that must be run and pass — and what the null looks like

Define one headline quantity: **selectivity** = per-event log-score gain of the full model over the *strongest* matched control, measured out of fold at the fitted mixture weight, reported as the **resolution** component, on a person-disjoint arm and a time-respecting arm, with person-level cluster bootstrap CIs and Holm correction across the comparison family.

**E1 — Channel decomposition.** Four numbers per dataset: observed-only (β=1), outcome-only, both at β=1, both at out-of-fold λ̂. **β=1 is the headline; the β sweep is sensitivity, never the reverse.** *Null:* outcome-only ≈ marginal/popularity baseline and both ≈ observed-only. (Your Table 1 says this is already the outcome.)

**E2 — Control ladder**, all arms matched on K, token length, embedding norm and fitted weight: (a) real consequence narratives; (b) deterministic template of exactly the recorded attributes; (c) **descriptive narratives** — same LLM, same length, present-tense description, no consequences (this arm isolates *prospective* from *descriptive* and is the genuinely unoccupied comparison); (d) attributes permuted across alternatives within the choice set, same generator; (e) narratives from a matched alternative in the same category; (f) random words; (g) matched-norm random embeddings; (h) uniform floor at λ̂; (i) marginal/climatology floor at its own optimal weight. Report selectivity. *Null:* (b), (c) or (d) recovers most of the gain — which your template result and shuffled-sentence result already indicate.

**E3 — Conditional independence test.** H0: choice ⊥ narrative | observed attributes, on held-out data, via the Holdout Randomization Test (Tansey et al., JCGS 2021) or Conditional Predictive Impact (Watson & Wright, ML 2021) for a signed effect size with a CI in log-likelihood units. This is the only experiment that converts the disputed ΔNLL table into a p-value. *Null:* fail to reject on all four datasets → the channel carries nothing beyond the attributes, and that is the paper's finding.

**E4 — Erasure.** LEACE the embeddings against the within-set attribute-difference matrix; report erased vs unerased; sweep erasure rank with R-LACE for a rank-vs-gain curve. Scope the claim to linear (or a stated kernel) recoverability — Kernelized Concept Erasure (EMNLP 2022) shows exhaustive nonlinear erasure is unsolved. *Null:* gain collapses at rank 1-2 → the entire contribution lives in the attribute span.

**E5 — Score decomposition and calibration competitors.** Reliability/resolution/uncertainty decomposition of the log score; temperature-scale the base model alone on the same out-of-fold data (Guo et al. 2017); beta-transformed linear pool (Ranjan & Gneiting 2010). *Null:* the gain is entirely reliability, and temperature scaling of the base model alone recovers it.

**E6 — Exclusion-restriction arm.** `A_gen` / `A_obs` partition as above. *Null:* performance ≤ observed-only with all attributes → no information beyond a feature transformation.

**E7 — Baselines at documented equal tuning budget** (identical search-space size, trial count, seeds, in a table): tuned conditional logit with the same amplification; L-MNL; RUMBoost; AlphaRec-style linear probe on the same frozen embeddings; plain 2-layer MLP on the same embeddings; BC-LLM; persona-embedding method on Swissmetro; zero-shot LLM ranker. *Null:* already observed — the plain MLP matches or beats the designed decomposition on every dataset, so the factored architecture (semantic scores × simplex person weights × salience) is unjustified and must be demoted or dropped.

**E8 — Protocol repairs.** Report both a within-person chronological arm and a **person-disjoint arm** (w(z) and the salience head both take person covariates, so the architecture has two explicit channels for memorising customer identity). Global-timeline check or time-truncated popularity feature — your ΔNLL reference baseline is currently computed from pooled-training-split popularity, which is future information relative to some test events, so a contaminated zero contaminates every uplift. Full-catalogue evaluation on an event subsample or the Krichene–Rendle corrections. Contamination test on Amazon (Bordt et al. 2024). Variance decomposition over split-strategy × seed × hyperparameter draw (Bouthillier et al. 2021) instead of three seeds at a fixed configuration.

**E9 — Intervention test for generated bottlenecks** (the one genuinely new faithfulness protocol available): edit a narrative to negate one semantic axis; check the axis score and V_out move in the predicted direction; report hit rate against a random-edit control. *Null:* hit rate ≈ chance → the axes are labels, not a bottleneck; the interpretability claim must be withdrawn.

**E10 — Human elicitation** on a few hundred events (highest value, only if affordable): ask people to list the consequences they considered; measure agreement with generated narratives. This is the only evidence that makes the word "perceived" defensible. *Null:* low agreement → rename the construct to what is demonstrated, a structured semantic expansion channel.

**The overall null, stated plainly:** if E3 fails to reject everywhere and E2 selectivity CIs include zero, the honest paper is *"generated consequence narratives carry no choice-relevant information beyond the attributes they were generated from; here is the protocol that establishes it and the decomposition that explains why prior positive reports in this family are mixing and calibration artefacts."* That paper is publishable. The current one is not.

---

## D. Must-cite and must-differentiate-from

**Choice modelling — the omission that produced objections 1 and 2:**
- *Hybrid Choice Models: Progress and Challenges* (Ben-Akiva et al., Marketing Letters 2002)
- *Generalized random utility model* (Walker & Ben-Akiva, Math. Social Sciences 2002)
- *How, when and why integrated choice and latent variable models are latently useful* (Vij & Walker, TR-B 2016) — **the single most dangerous uncited paper; concede it explicitly**
- *About attitudes and perceptions: finding the proper way to consider latent variables in discrete choice models* (Bahamonde-Birke et al., Transportation 2017)
- *Practical and empirical identifiability of hybrid discrete choice models* (Raveau et al., TR-B 2012)
- *Sequential and Simultaneous Estimation of Hybrid Discrete Choice Models: Some New Findings* (Raveau et al., TRR 2010)
- *On the (im-)possibility of deriving transport policy implications from hybrid choice models* (Chorus & Kroesen, Transport Policy 2014)
- *Subjective variables in travel behavior models: a critical review and Standardized Transport Attitude Measurement Protocol (STAMP)* (Bhagat-Conway et al., Transportation 2024) — your strongest positive motivation
- *Using ordered attitudinal indicators in a latent variable choice model* (Daly et al., Transportation 2012) — the measurement-equation template
- *Enhancing discrete choice models with representation learning* (Sifringer, Lurkin & Alahi, TR-B 2020) — L-MNL, the two-channel precedent and required baseline
- *Can large language models assist choice modelling? Insights into prompting strategies and current models' capabilities* (Sfeir et al., J. Choice Modelling 2026)

**LLM-generated features / text as features — the prior art that closes the architecture:**
- *Synthesizing Attitudes, Predicting Actions (SAPA)* (arXiv:2509.18181)
- *Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning* (ATHENA, arXiv:2511.02194)
- *TabLLM: Few-shot Classification of Tabular Data with Large Language Models* (AISTATS 2023)
- *Waffling around for Performance: Visual Classification with Random Words and Broad Concepts* (ICCV 2023)
- *Visual Classification via Description from Large Language Models* (ICLR 2023)
- *Learning Concise and Descriptive Attributes for Visual Recognition* (ICCV 2023)
- *End-to-end Training for Recommendation with Language-based User Profiles* (LangPTune, arXiv:2410.18870)
- *Language Representations Can be What Recommenders Need: Findings and Potentials* (AlphaRec, ICLR 2025 Oral)
- *Elephants Never Forget: Memorization and Learning of Tabular Data in Large Language Models* (COLM 2024)
- *From Residuals to Reasons: LLM-Guided Mechanism Inference from Tabular Data* (arXiv:2605.22897) — closest prior art to your two-stage numeric-residual design
- *Bayesian Concept Bottleneck Models with LLM Priors* (BC-LLM, NeurIPS 2025)

**Concept bottleneck / leakage — the vocabulary the objections are written in:**
- *Promises and Pitfalls of Black-Box Concept Learning Models* (Mahinpei et al. 2021)
- *Addressing Leakage in Concept Bottleneck Models* (Havasi, Parbhoo & Doshi-Velez, NeurIPS 2022)
- *Post-hoc Concept Bottleneck Models* (Yuksekgonul et al., ICLR 2023) — PCBM-h residual precedent
- *Quantifying the Accuracy-Interpretability Trade-Off in Concept-Based Sidechannel Models* (Debot et al. 2025) — the Sidechannel Independence Score
- *In Defense of Information Leakage in Concept-based Models* (Espinosa Zarlenga, ICML 2026) — lets you declare the attribute channel a side-channel and argue benign leakage, *if* you measure it
- *There Was Never a Bottleneck in Concept Bottleneck Models* (ICLR 2026) — cite where you fix §3.2
- *If Concept Bottlenecks are the Question, are Foundation Models the Answer?* (2025/2026)

**Method / protocol:**
- *LEACE: Perfect linear concept erasure in closed form* (NeurIPS 2023)
- *Log-linear Guardedness and its Implications* (ACL 2023)
- *Kernelized Concept Erasure* (EMNLP 2022) — the scope caveat
- *Double/debiased machine learning for treatment and structural parameters* (Econometrics Journal 2018)
- *The Holdout Randomization Test for Feature Selection in Black Box Models* (JCGS 2021); *Testing conditional independence in supervised learning algorithms* (Machine Learning 2021)
- *Designing and Interpreting Probes with Control Tasks* (EMNLP-IJCNLP 2019); *Information-Theoretic Probing with Minimum Description Length* (EMNLP 2020)
- *Combining Probability Forecasts* (Ranjan & Gneiting, JRSS-B 2010); *Optimal Prediction Pools* (Geweke & Amisano 2011); *Reliability, Sufficiency, and the Decomposition of Proper Scores* (Bröcker 2009); *Kullback-Leibler Divergence as a Forecast Skill Score…* (Weijs et al. 2010)
- *Using Stacking to Average Bayesian Predictive Distributions* (Yao et al. 2018) — concede the pooling step
- *Deep Ensembles Work, But Are They Necessary?* (NeurIPS 2022) — your framing precedent
- *Accounting for Variance in Machine Learning Benchmarks* (MLSys 2021); *On Sampled Metrics for Item Recommendation* (KDD 2020); *Leakage and the reproducibility crisis in machine-learning-based science* (Patterns 2023); *REFORMS* (Science Advances 2024)
- *A General Approach to Causal Mediation Analysis* (Imai, Keele & Tingley, Psych. Methods 2010) — cite it or drop "mediated"

---

## E. Defensible abstract for the revised paper

> Discrete-choice models are increasingly augmented with text generated by large language models, on the premise that generated narratives supply latent perceptual content that tabulated attributes omit. This premise is rarely tested: the standard evidence is an improvement in held-out log-likelihood, which the hybrid-choice literature has long held is not evidence of a latent construct, and which the forecasting literature shows is partly guaranteed by the arithmetic of mixing. We formalise outcome-conditioned utility as a generalised random-utility model in which a frozen language model is an imperfect proposal distribution over an unobserved consequence space, with an explicit structural equation, an explicit measurement equation, and stated normalisations. We then supply the identification protocol the formulation requires: a ladder of information-matched controls transported into random-utility modelling for the first time (a deterministic template over exactly the recorded attributes, descriptive rather than prospective narratives at matched length and count, narratives generated from attributes permuted within the choice set, random text, and matched-norm random embeddings), a weight-matched uninformative-channel control that holds the fitted mixture weight fixed and substitutes an uninformative distribution — bounding what mixing alone can buy, which to our knowledge has no counterpart in the pooling or probing literature — a closed-form linear erasure of the narrative embeddings against the span of within-choice-set attribute differences, a holdout conditional-independence test of choice against narrative given the observed attributes, and a reliability–resolution decomposition of the log score that separates calibration gain from information gain. Applied to four choice datasets under respondent-disjoint and time-respecting splits with out-of-fold mixture weights, the protocol returns a consistent negative result: the generated narrative channel is not distinguishable from a template over the same attributes, an amplified observed-attribute side-channel accounts for the headline gains reported under the conventional protocol, a parameter-matched channel containing no text attracts a larger fitted weight than the text channel, and on two datasets the entire apparent improvement is the mixing bound rather than information. Under an exclusion restriction in which a designated attribute subset is withheld from the generator, we report the residual contribution that survives erasure, scoped to linear non-recoverability. We conclude that claims that generated text carries perceived-outcome information beyond the attributes it was generated from require this protocol rather than a likelihood comparison, and we release prompts, model versions, decoding parameters and the content-addressed narrative cache so that every reported number is exactly reproducible.

---

**Bottom line.** The paper cannot be saved by rebuttal — three of four objections are confirmed by your own experiments and the fourth is a work item. It can be saved by relocating the claim: the architecture is not the contribution, the identification protocol and the negative finding are, with the exclusion-restriction variant as the constructive remainder. Fix the §3.2 containment claim, drop or defend "mediated", add the ICLV literature, and remove the injected text before anything else.