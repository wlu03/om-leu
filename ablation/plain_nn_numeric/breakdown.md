**Problem:**  
The same plain MLP with the sentences removed: $u_{ij} = \mathrm{MLP}([z_i; \tilde x_{ij}; \mathbb{1}_j; h_{ij}])$. Why include a row that uses neither the LLM nor the designed model?

**Explanation:**  
The 2×2 design behind `ablation/` crosses *LLM sentences: no / yes* with *designed model: no / yes*. `no_sentences` is (no, yes) on the structural side, `full_model` is (yes, yes), the sentence MLPs are (yes, no), and this folder is (no, no) for an unstructured learner. Without it the value of the sentences to a *plain* network could not be isolated, because `plain_nn_all_inputs` mixes the numeric information with the sentence information.

The two differences

$$
\mathrm{NLL}(\texttt{plain\_nn\_numeric}) - \mathrm{NLL}(\texttt{plain\_nn\_all\_inputs}) \qquad\text{and}\qquad \mathrm{NLL}(\texttt{no\_sentences}) - \mathrm{NLL}_{\text{mix}}(\texttt{full\_model})
$$

are the value of the sentences to an unstructured model and to the designed model respectively. If the first is large and the second small, the sentences carry information that the structural model already extracts from the numbers (the sentences restate $x_{ij}$ in words, as the shuffled-sentence controls in `docs/llm_signal_research.md` suggest). If both are of similar size, the sentences add something orthogonal to the numeric attributes that every learner benefits from. If the first is small and the second large, the design is needed to make the sentences useful.

This row also reproduces, inside the ablation harness, the finding of `methods/results/comparison_table.md` that an unconstrained network on the numeric attributes is a strong model on Optima and LPMC. Its sentence-only column is the numeric MLP alone; its mixture column mixes that MLP with the structural channel, an ensemble of two numeric models, which is reported only for completeness.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
