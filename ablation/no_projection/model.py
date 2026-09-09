"""No low-rank projection: heads act on the LayerNormed 768-d embedding directly."""
NAME, GROUP = "no_projection", "design"
LABEL = "no 768 -> 32 projection (heads on the full embedding)"
CONFIG = dict(member_kw=(("proj", False),))
