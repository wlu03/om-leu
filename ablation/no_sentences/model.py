"""No LLM sentences at all: stage 1 + stage 2 with the temperature, members = 0."""
NAME, GROUP = "no_sentences", "baseline"
LABEL = "no sentences: structural utility + boosted residual + temperature (no LLM input)"
CONFIG = dict(members=0)
