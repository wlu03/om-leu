"""E1 grounded generation: prompt, allowlist and output schema.

The grounded prompt asks for the favourable and adverse implications that the supplied
information supports, each tagged with its evidence status, and forbids invented precision
(parking availability, delay distributions), affordability judgements and demographic
preference stereotypes.  Outcome-relevant personalisation is allowed - carrying shopping,
holding a season ticket - because those are recorded facts; inferred liking is not, and
belongs in the valuation model.
"""
from __future__ import annotations

import json
from typing import Dict, List, Mapping, Sequence

from omleu_experiments.contracts import AXES, EVIDENCE_STATUS

#: Pre-choice fields a generator may see.  The two person aggregates that summarise the
#: respondent's own choices (novelty_rate, purchase_frequency) are excluded: they are
#: derived from labels and from the respondent's later events.
GROUNDED_ALLOWLIST = (
    "alternative_name", "travel_time", "cost", "currency", "cost_note", "waiting_time",
    "access_time", "transfers", "distance", "seats_or_class",
    "age_bucket", "income_bucket", "household_size", "has_kids", "education", "area_type",
    "licence", "car_availability", "season_ticket", "trip_purpose", "trip_distance", "origin_area",
    "prior_choices_of_this_alternative",     # strictly earlier events only
)

PERSON_FREE_ALLOWLIST = tuple(f for f in GROUNDED_ALLOWLIST if f in (
    "alternative_name", "travel_time", "cost", "currency", "cost_note", "waiting_time",
    "access_time", "transfers", "distance", "seats_or_class", "trip_purpose", "trip_distance"))

GENERATION_SPEC = {
    "schema": {"axis": "one of " + ", ".join(AXES),
               "text": "one first-person sentence",
               "evidence_status": "one of " + ", ".join(EVIDENCE_STATUS),
               "supporting_fields": "list of field names from the allowlist",
               "assumptions": "list of stated rules used for a derived claim",
               "missing_information": "list of quantities the evidence does not establish"},
    "decoding": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 400, "seed_role": "generation"},
}

SYSTEM = (
    "You describe what would follow for one person if they took one travel option. "
    "You may only use the supplied fields. For every sentence you must mark its evidence "
    "status: 'observed' if it restates a supplied value, 'derived' if it follows from a "
    "supplied value under a rule you state, and 'unknown' if the evidence does not "
    "establish it. Never invent a quantity that was not supplied, never state parking "
    "availability, delay probabilities or crowding unless a field gives them, never judge "
    "whether an amount is cheap, negligible or affordable, and never infer a preference "
    "from age, income, sex or occupation. Write one sentence per axis."
)

USER = """Option: {alternative_name}
Supplied fields (these are the only facts you have):
{fields}

Write exactly {K} first-person sentences, one for each axis in this order: {axes}.
Return JSON: a list of objects with keys axis, text, evidence_status, supporting_fields,
assumptions, missing_information.
For a quantity that is not supplied, write the sentence with evidence_status "unknown" and
name the missing quantity, for example: "The evidence does not establish the parking cost
at my destination."
"""


def grounded_messages(fields: Mapping[str, object], alternative_name: str, K: int = 5) -> List[Dict[str, str]]:
    rendered = "\n".join(f"- {k}: {v}" for k, v in sorted(fields.items()) if v is not None)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(alternative_name=alternative_name, fields=rendered,
                                                    K=K, axes=", ".join(AXES))}]
