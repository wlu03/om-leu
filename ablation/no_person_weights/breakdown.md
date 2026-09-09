**Problem:**  
Replace the person weights $w(z_i) = \operatorname{softmax}(\mathrm{MLP}_w(z_i))$ by the uniform vector $w_m = 1/M$, so that $\tilde V_{ij} = \frac{1}{M}\sum_m A_{ijm}$. What exactly is removed, and how should the loss be read against the plain-MLP baselines?

**Explanation:**  
In the designed member the covariates $z_i$ enter in one place only, the weights over the $M$ attribute heads (the salience attention used by the final model is person-agnostic). Setting the weights to $1/M$ therefore removes *all* person information from the sentence channel, not just the heterogeneity of tastes: the member becomes a function of the sentences alone,

$$
\tilde V_{ij} = \bar a^{\top} h_{ij} + \bar b, \qquad \bar a = \tfrac{1}{M}\sum_m a_m,\; \bar b = \tfrac{1}{M}\sum_m b_m,
$$

and the $M$ heads collapse into a single linear head on the pooled sentence, since a uniform average of linear maps is a linear map. Two things are tested at once: whether the sentence channel needs $z_i$, and whether it needs several heads (the second is isolated in `single_head`, which keeps $z_i$ trivially but has nothing to weigh).

The mechanism that is lost is the bilinear interaction. With person weights the utility is

$$
\tilde V_{ij} = \sum_m w_m(z_i)\, A_{ijm} = w(z_i)^{\top} A_{ij},
$$

a person-specific linear functional of the vector of attribute values; two people reading the same five sentences can rank the alternatives differently because they weight "lower monthly costs" and "more reliable arrival" differently. This is the sentence-channel counterpart of the TasteNet tastes $\beta_t(z_i), \beta_c(z_i)$ of stage 1: the heads play the role of attributes and the weights the role of tastes, with the difference that the attributes are read from text.

Reading the numbers: the sentence-only $\Delta$ against `full_model` is the value of $z_i$ inside the sentence model; the same quantity for an unstructured learner is `plain_nn_sentences_person` minus `plain_nn_sentences`. If the first is much larger than the second, the head-and-weight form is what makes the covariates usable. The mixture $\Delta$ will be smaller than the sentence-only $\Delta$, because the structural channel already carries $z_i$ through its tastes and intercepts and the fitted $\pi$ shrinks toward it; the sentence-only column is where the design is judged.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
