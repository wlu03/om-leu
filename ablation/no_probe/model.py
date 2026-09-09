"""No InfoNCE probe on the mean projected sentence during member pretraining."""
NAME, GROUP = "no_probe", "design"
LABEL = "no InfoNCE probe loss during pretraining"
CONFIG = dict(member_kw=(("probe", False),))
