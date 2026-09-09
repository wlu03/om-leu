"""No sentences, no structure: [z_i, x_ij, alt one-hot, h_ij] -> MLP (the unstructured no-LLM control)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import _plain_nn  # noqa: E402,F401
NAME, GROUP = "plain_nn_numeric", "baseline"
LABEL = "plain MLP on [z_i, x_ij, alt id, h_ij] only (no sentences, no structure)"
CONFIG = dict(member_kind="plain_nn", member_kw=(("inputs", ("z", "x")),))
