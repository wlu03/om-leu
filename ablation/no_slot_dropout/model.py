"""No sentence-slot dropout during member pretraining."""
NAME, GROUP = "no_slot_dropout", "design"
LABEL = "no sentence-slot dropout"
CONFIG = dict(member_kw=(("slot_drop", 0.0),))
