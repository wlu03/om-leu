"""Same LLM sentences plus the person covariates, plain MLP: [mean embedding, z_i] -> MLP."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import _plain_nn  # noqa: E402,F401
NAME, GROUP = "plain_nn_sentences_person", "baseline"
LABEL = "plain MLP on [mean sentence embedding, person covariates z_i]"
CONFIG = dict(member_kind="plain_nn", member_kw=(("inputs", ("sent", "z")),))
