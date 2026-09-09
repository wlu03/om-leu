"""Person-conditioned attention (query from z_i) instead of person-agnostic salience."""
NAME, GROUP = "person_attention", "design"
LABEL = "person-conditioned attention over the K sentences instead of salience"
CONFIG = dict(member_kw=(("attn", "person"),))
