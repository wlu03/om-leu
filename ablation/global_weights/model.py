"""Sentence model with one learned weighting of the five heads, identical for every person.

This separates two things that the `no_person_weights` ablation removes together: the
model's ability to learn *which* topics matter, and its ability to learn that *per person*.
Uniform 1/M removes both; a learned global vector removes only the second.
"""
NAME, GROUP = "global_weights", "design"
LABEL = "one learned head weighting shared by all persons (no personalisation, but not uniform)"
CONFIG = dict(member_kw=(("weights", "global"),))
