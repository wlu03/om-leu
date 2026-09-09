"""One attribute head instead of M = 5 (the person weights then have nothing to weigh)."""
NAME, GROUP = "single_head", "design"
LABEL = "single attribute head (M = 1; no head decomposition, no person weighting)"
CONFIG = dict(member_kw=(("M", 1),))
