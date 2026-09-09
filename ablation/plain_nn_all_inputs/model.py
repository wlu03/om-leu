"""Everything the full model sees, no structure: [mean embedding, z_i, x_ij, alt one-hot, h_ij] -> MLP."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import _plain_nn  # noqa: E402,F401
NAME, GROUP = "plain_nn_all_inputs", "baseline"
LABEL = "plain MLP on [mean sentence embedding, z_i, x_ij, alt id, h_ij] (all inputs, no structure)"
CONFIG = dict(member_kind="plain_nn", member_kw=(("inputs", ("sent", "z", "x")),))
