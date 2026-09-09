"""Same LLM sentences, plain MLP: mean of the K sentence embeddings -> MLP(128, 64) -> utility."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import _plain_nn  # noqa: E402,F401
NAME, GROUP = "plain_nn_sentences", "baseline"
LABEL = "plain MLP on the mean sentence embedding (no structure, no person input)"
CONFIG = dict(member_kind="plain_nn", member_kw=(("inputs", ("sent",)),))
