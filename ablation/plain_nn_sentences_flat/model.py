"""Same LLM sentences, plain MLP on the K embeddings concatenated (slot identity kept, 3840-d input)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import _plain_nn  # noqa: E402,F401
NAME, GROUP = "plain_nn_sentences_flat", "baseline"
LABEL = "plain MLP on the K concatenated sentence embeddings (slot order kept)"
CONFIG = dict(member_kind="plain_nn", member_kw=(("inputs", ("sent_flat",)),))
