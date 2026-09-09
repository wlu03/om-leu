"""Sentence model without personalised head weights: w_m(z_i) replaced by the uniform 1/M."""
NAME, GROUP = "no_person_weights", "design"
LABEL = "no personalised head weights (uniform 1/M instead of w_m(z_i))"
CONFIG = dict(member_kw=(("weights", "uniform"),))
