**Problem:**  
The model is run with no sentence channel at all: stages 1–2 with the temperature $a$ and $M = 0$ members, so $p_i(j) = \operatorname{softmax}_j(a\,U_{ij})$. What does this row fix as the reference, and why is its mixture NLL the right zero point for "how much comes from the LLM"?

**Explanation:**  
With no members the mixture collapses to the structural channel alone,

$$
p_i(j) = q_i(j) = \operatorname{softmax}_j\!\big(a\,U_{ij}\big), \qquad U_{ij} = \tau L_{ij} + f_j(x_{ij}, z_i, h_{ij}),
$$

and only the temperature is fitted on validation. Every other row adds a sentence channel to exactly this $q_i$, so

$$
\Delta_{\text{LLM}}(\text{row}) = \mathrm{NLL}(\text{row}) - \mathrm{NLL}(\texttt{no\_sentences})
$$

is the value of that row's sentence channel *given* the structural model. It is the quantity the paper claims, and it is bounded by the mixture identity

$$
\log\!\big((1-\pi) q_i(y_i) + \pi \bar p_i(y_i)\big) - \log q_i(y_i) = \log\!\Big(1 + \pi\big(\tfrac{\bar p_i(y_i)}{q_i(y_i)} - 1\big)\Big),
$$

which is positive only on events where the sentence channel puts more mass on the chosen alternative than the structural model does. The row therefore separates two things the plain-MLP baselines cannot separate on their own: how much the *sentences* add (compare `plain_nn_sentences` mixed with the structural model against this row) and how much the *designed sentence model* adds on top (compare `full_model` against `plain_nn_sentences`). The temperature is kept so that the comparison is not confounded by calibration: a mixture with $\pi \to 0$ and a temperature-only model are the same model, which is why the reference's $\pi$ is reported next to every mixture NLL.

The sentence-only column is empty for this row because there is no sentence channel to evaluate.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
