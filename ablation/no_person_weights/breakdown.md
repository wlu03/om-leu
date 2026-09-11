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

The sentence-only $\Delta$ against `full_model` is the value of $z_i$ inside the sentence model; the same quantity for an unstructured learner is `plain_nn_sentences_person` minus `plain_nn_sentences`. If the first is much larger than the second, the head-and-weight form is what makes the covariates usable. The mixture $\Delta$ will be smaller than the sentence-only $\Delta$, because the structural channel already carries $z_i$ through its tastes and intercepts and the fitted $\pi$ shrinks toward it; the sentence-only column is where the design is judged.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.

---

**Problem:**
Removing the personalised weights costs up to $0.10$ nats when the sentence channel predicts
alone and essentially nothing inside the mixture. Explain the size of that gap.

**Explanation:**
Four measurements settle it (`ablation/why_no_change.py`, `ablation/dilution.py`; Optima and
LPMC, seed 7, person-level split, out-of-fold $\pi$).

*The personalisation is genuine.* The learned weights vary across people: on LPMC the
reliability head's weight has standard deviation $0.23$ and range $0.92$, and only $65\%$ of
people share a top-weighted head. A third variant that learns **one** weight vector shared by
everyone (`ablation/global_weights/`) recovers none of the loss: $0.5626$ against $0.5628$ for
fixed uniform weights and $0.5262$ for the personalised version, with learned weights that are
nearly flat. So what the ablation removes is per-person allocation, not the ability to learn
which axes matter.

*The standalone loss is a tail, not a shift.* Sorting test events by how much the ablation hurts
the sentence channel, the top decile carries a mean loss of $+0.972$ on Optima and $+0.533$ on
LPMC, while the remaining nine tenths carry $+0.052$ and $-0.018$: on LPMC the ablated model is
slightly **better** on ninety per cent of events. The mean standalone loss is produced by a small
set of events on which the uniform-weight model fails badly.

*Those events are ones the structural model already handles.* The structural model's arg-max is
already the chosen alternative on $72\%$ of Optima's tail decile and $67\%$ of LPMC's.

*The mixture bounds what any component failure can cost.* Since
$p_i(y) = (1-\pi) s_i(y) + \pi q_i(y) \ge (1-\pi) s_i(y)$,

$$
-\log p_i(y) \;\le\; -\log s_i(y) \;-\; \log(1-\pi),
$$

so however badly the semantic channel fails on an event, the mixture loses at most
$-\log(1-\pi)$ nats relative to the structural channel alone: $0.21$ at $\pi = 0.19$ and $0.38$
at $\pi = 0.32$. Standalone the same failure is unbounded, which is exactly what the tail decile
shows. The first-order sensitivity tells the same story from the other side: the change in the
mixture loss is the standalone change multiplied by the responsibility $\pi q_i(y)/p_i(y)$,
and on precisely the events where the ablated channel collapses, $q_i(y)$ collapses with it, so
the responsibility goes to zero and the mixture stops listening.

Measured attenuation, standalone change divided by the change in the mixture: $25.9\times$ on
Optima and $21.9\times$ on LPMC. Refitting $\pi$ accounts for part of it on Optima, where
$\pi$ falls from $0.188$ to $0.143$ and the damage halves from $0.0132$ to $0.0055$, and for
none of it on LPMC, where $\pi$ barely moves.

The implication for the design is specific. The personalised weights raise the sentence
channel's average accuracy but do not change *which* events it wins: its win rate against the
structural channel is $36.2\%$ before the ablation and $36.1\%$ after it on LPMC. A component
that improves accuracy without changing the set of events it wins is exactly what a probability
mixture discards.
