"""Sentence model without the salience mechanism: mean pooling over the K sentences."""
NAME, GROUP = "no_salience", "design"
LABEL = "no salience attention (mean over the K sentences)"
CONFIG = dict(member_kw=(("attn", "mean"),))
