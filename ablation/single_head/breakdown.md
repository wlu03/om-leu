**Problem:**  
Use one attribute head instead of $M = 5$: $A_{ij} = a^{\top} h_{ij} + b$, so that $w(z_i)$ is a softmax over a single entry, equal to 1, and $\tilde V_{ij} = A_{ij}$. How does this differ from `no_person_weights`, and what does the pair of ablations identify?

**Explanation:**  
Both ablations reduce the member to a single linear head on the pooled sentence, so both lose the bilinear person × attribute interaction. They differ in what they keep: `no_person_weights` keeps five heads and averages them (which is equivalent to one head, as shown there), while `single_head` keeps the person-weight network but gives it nothing to choose between. The two rows should therefore agree up to optimisation noise. If they do, the head decomposition has no value *by itself*; its value is entirely in giving the person weights something to weigh, and the loss in both rows is the loss of person-specific weighting of sentence-derived attributes.

If they do not agree, the difference is informative. `no_person_weights` with five heads still trains five different linear maps during pretraining before averaging them, so it can behave like a small within-member ensemble; if it beats `single_head`, that averaging is worth something on its own. Conversely, a `single_head` member has fewer parameters and may early-stop later; if it beats `no_person_weights`, the five heads were overfitting once their weights stopped depending on the person.

The number $M = 5$ was chosen to match the five consequence axes, so that each head can be interpreted as reading one attribute (the head-alignment analysis in the OM-LEU pipeline pursues that interpretation). The ablation does not test interpretability; it tests whether the decomposition is needed for prediction. Together with `no_person_weights` it answers: the heads are needed exactly as much as the person weights are, and neither is useful without the other.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
