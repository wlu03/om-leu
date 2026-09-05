# Architectural ideas transplantable into OM-LEU — literature survey (2026-09-05)

Modules referenced: OutcomeGen (frozen LLM, K = 5 sentences per person × alternative) →
Enc (frozen all-mpnet-base-v2) → Heads (M = 5) → WeightNet (softmax over heads from
person covariates) → SalienceNet (weights over K sentences) → Residual (linear MNL on
numeric attributes) → softmax over alternatives. Items marked [unverified] could not be
checked against the full text.

## 1. Fusing text / LLM embeddings with tabular features
- `e-mnl` — Arkoudi, Krueger, Azevedo, Pereira (2023), *TR-B* 175:102783, arXiv:2109.12042.
  Categorical variables → embeddings with one dimension per alternative, so coordinates read
  as alternative-specific coefficients. Transplant: alternative-indexed embeddings for the
  categorical person covariates feeding WeightNet / Residual. Cost low.
- `llm-emb-tabular` — Koloski et al. (2025), arXiv:2502.11596; Kolomenko et al. (2026),
  arXiv:2603.17737. Serialise each feature separately, embed with a frozen encoder,
  *concatenate* to the raw numbers (never replace them). +3 pp accuracy avg over 7 datasets;
  the only loss was on an all-numeric set, attributed to weak numeric representation.
  Transplant: per-attribute sentences in Enc; keep Residual next to the embeddings.
- `tp-berta` — Yan et al. (2024), ICLR, arXiv:2403.01841. Relative Magnitude Tokenization:
  binned magnitude token embedding × raw value, magnitude-aware triplet loss, intra-feature
  attention. Value-as-string costs 12.45 % AUC in their ablation. Transplant: a magnitude
  channel concatenated to the mpnet embedding before Heads, with the triplet regulariser.
- `carte` — Kim, Grinsztajn, Varoquaux (2024), ICML, arXiv:2402.16785. Row as a star graphlet
  of (column name, value) with name embedding × value; pretrained on YAGO. Transplant:
  graphlet encoder of an alternative's attributes instead of one sentence embedding. Cost
  medium-high.
- `mo-llm-travel` — Mo et al. (2023), arXiv:2312.00819. LLM-only baselines on Swissmetro
  (GPT-3.5 zero-shot 0.586 vs MNL 0.605, NN 0.635): the sanity anchor for any semantic branch.

## 2. Numeric-aware encoders
- `xval` — Golkar et al. (2023), arXiv:2310.02989. A single [NUM] token whose embedding is
  multiplied by the value; end-to-end continuous in numbers. Cheap variant: add x_j · e_j
  after pooling with learned directions e_j. See also Zhang et al. (2020), Findings of EMNLP,
  arXiv:2010.05345 (off-the-shelf embeddings capture only coarse scale).
- `fone` — Zhou et al. (2025), arXiv:2502.09741. Fourier-feature number embeddings; exact,
  single-token. Transplant: Fourier features of cost / time appended to the head input.
  Evidence is on arithmetic, so treat as an ablation.

## 3. Structured / interpretable heads
- `rumnet` — Aouad, Désir (2025), *Management Science*, arXiv:2207.12877. Utility as a neural
  function of features plus latent type noise; averaged softmax; provably approximates any RUM.
  Transplant: WeightNet as a latent-type mixture (T× forward cost).
- `tastenet-mnl` — Han, Pereira, Ben-Akiva, Zegras (2022), *TR-B*, arXiv:2002.00922.
  Person covariates → sign-constrained taste parameters of a linear MNL. Transplant: WeightNet
  also outputs Residual's coefficients β(z). See also Vallarino (2025), arXiv:2503.05800 (MoE
  gating on consumer covariates) [no numbers].
- `conjoint-structural-dl` — Acharya, Hainmueller, Xu (2026), arXiv:2604.10845. DNN
  preference vectors from respondent characteristics with cross-fitting and debiased-ML
  inference. Transplant: cross-fit WeightNet / β(z) and report population-average head
  weights with standard errors.
- `label-free-cbm` — Oikarinen et al. (2023), ICLR, arXiv:2304.06129 (on Koh et al. 2020;
  sparse CBM arXiv:2404.03323). LLM-proposed named concepts, bottleneck neurons aligned to
  text-embedding anchors (cos³ loss), sparse linear readout. Transplant: replace the free
  heads by concept scores against fixed named anchors ("I arrive late", "it strains my
  budget"); head collapse disappears by construction. Vision figures [from memory].

## 4. LLM utilities from log-probabilities / preference models
- `innate-econ-prefs` — Buchanan, Foster (2026), arXiv:2607.26288; Mazeika et al. (2025),
  NeurIPS, "Utility Engineering". A single-token answer over a menu *is* a conditional logit:
  label logits are a utility index. Transplant: permutation-averaged label logits from an
  open-weight LLM as a fixed feature q into a Stage-2 correction g(q). One forward pass per
  choice set (vs K generations per alternative). IIA / reflexivity failures are position-bias
  driven; average over permutations.
- `bt-regression` — Sun, Shen, Ton (2025), ICLR oral, arXiv:2411.04991. Reward models are BT
  regression on frozen LLM embeddings; any order-preserving objective suffices. Transplant:
  pairwise chosen-vs-unchosen loss on Heads; LLM hidden states as head input.

## 5. Fine-tuning / adapting the encoder
- `encoderec` — Hadad, Rabaev, Shapira (2026), arXiv:2601.10837. In-batch InfoNCE adaptation
  of a small encoder to item metadata; 6–62 % Recall@10 gains under UniSRec. Transplant:
  contrastive fine-tune of mpnet with anchor = person context, positives = chosen
  alternative's sentences, negatives = unchosen; person-level splits needed.
- `litransmc` — Alsaleh, Farooq (2025), arXiv:2507.21432. LoRA with answer-token loss masking
  on a 12B model beats untuned locals and GPT-4o (wF1 0.6845, JSD 0.000245). Transplant:
  LoRA-tune OutcomeGen on the choice token, or use its choice logits as the q feature.

## 6. Two-stage / residual designs with structural guarantees
- `fm-adapter` — Wang, Sun, Li, Fan, Zhuang (2026), arXiv:2606.26432 (workshop
  arXiv:2605.26559). V_k = βᵀφ_k(x) + g_k(q(x)); Stage 1 fits β with g ≡ 0, Stage 2 freezes β;
  MRS preserved exactly (Prop. 1); joint training breaks identifiability when cost is
  recoverable from q (Prop. 2). Swissmetro / LPMC: +6.4 pp avg accuracy over MNL, VOT 84.4
  CHF/h preserved vs 1,509 for a jointly trained augmented MNL, 100 % monotone. This is the
  design `docs/omleu_improvements.md` implemented.
- `l-mnl-reslogit` — Sifringer, Lurkin, Alahi (2020), *TR-B* 140; Wong, Farooq (2021), *TR-C*
  126, arXiv:1912.10058. Data separation (the learned term sees only variables excluded from
  the linear part) and zero-initialised residual blocks. Transplant: orthogonalise head outputs
  against the numeric attribute vector so Heads explain only what Residual cannot.
- `gradient-reg` — Feng et al. (2024), *TR-C* 166, arXiv:2404.14701. Sign penalties on
  ∂P/∂x; +20 pp behavioural regularity and +1.7 % LL at small n. Transplant: once a magnitude
  channel makes Heads differentiable in cost / time, add the same penalty through the whole
  forward.

## 7. Counterfactual / consistency training for LLM features
- `athena` — Zhao et al. (2025), arXiv:2511.02194. LLM-driven symbolic utility search +
  per-person TextGrad templates. Transplant: one outcome sentence per fitted utility term, so
  K = number of terms and each sentence keeps its numeric anchor. Evidence on a 500-traveller
  subsample with an unusually weak MNL [suggestive only].
- `featllm` — Han, Yoon, Arik, Pfister (2024), ICML, arXiv:2404.09491. LLM proposes
  dataset-level rules once; rules are executed in code as features; ~10 % gain over TabLLM in
  few-shot regimes. Risk note: "LLM-Derived Preference Judgments Are Not Self-Consistent"
  (2026), arXiv:2608.17644 (41.7–87.5 % disagreement across framings). Transplant:
  LLM-written rules over person × attribute evaluated exactly and added to Residual.

## 8. Calibration
- `llm-calibration` — Wang et al. (2024), arXiv:2410.06707 (invert-softmax + temperature);
  Cao et al. (2025), NAACL, arXiv:2502.07068 (first-token-distribution loss for survey
  shares); arXiv:2510.21977; arXiv:2605.27752 (protocol sensitivity). Transplant: any LLM
  signal enters at logit scale through a learned temperature; report ECE with LL.

## Ranked top-5 (agent's ranking)
1. Two-stage residual training (`fm-adapter`) — zero cost, strongest evidence on these datasets.
2. Concept-bottleneck heads (`label-free-cbm`) — removes the collapse mechanism.
3. Magnitude channel (`tp-berta` / `xval` / `fone`) — documented cause and fix for blurred numbers.
4. Person-conditional tabular coefficients (`tastenet-mnl` + cross-fitting).
5. LLM choice logits as a fixed, calibrated Stage-2 feature (`innate-econ-prefs` + `fm-adapter`).
Runner-up: contrastive encoder fine-tuning (`encoderec`).
