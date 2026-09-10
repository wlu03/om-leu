**Problem:**  
The mean-pooled sentence MLP has no person input. Add the standardised covariates: $u_{ij} = \mathrm{MLP}([\phi_{ij}; z_i])$. Why is this the right control for the designed model's person weights, and what would each outcome mean?

**Explanation:**  
In the designed sentence model the person enters only through the weights over the heads,

$$
\tilde V_{ij} = \sum_m w_m(z_i)\, A_{ijm}, \qquad w(z_i) = \operatorname{softmax}(\mathrm{MLP}_w(z_i)),
$$

a bilinear form in (person weights) × (sentence-derived attribute values). The ablation `no_person_weights` shows what removing $z_i$ from that model costs. This folder asks the converse question: given the same $z_i$, does an unstructured network recover that value? The MLP on $[\phi_{ij}; z_i]$ can represent any interaction between sentence content and person, including the bilinear one, but it must find it from data with no inductive bias, and the choice-set structure is weak here: $z_i$ is the same for every alternative in the set, so it changes the softmax only through its interaction with $\phi_{ij}$ in the hidden layers, never additively (an additive term in $z_i$ cancels in the softmax over $j$).

Three readings are possible. If this folder matches `full_model` sentence-only, the person weights are simply "use $z_i$", and the head decomposition is a convenient rather than a necessary form. If it improves on `plain_nn_sentences` but stays well above `full_model`, the person information is useful but the bilinear head form extracts it more efficiently, which is evidence for the design. If it does not improve on `plain_nn_sentences` while `no_person_weights` shows a large loss, the designed form is what makes $z_i$ usable at all in the sentence channel. The paired $\Delta$ columns give both comparisons on the same events.

Notation and the full sentence model are in `ablation/full_model/breakdown.md`; the paired ΔNLL and its bootstrap interval are defined in `docs/math/05_paired_evaluation.md`.
