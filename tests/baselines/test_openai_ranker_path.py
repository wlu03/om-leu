"""The LLM rankers must use token logprobs on OpenAI-compatible endpoints.

Before this path existed, an ``OpenAILLMClient`` fell through to the
generic ``generate()`` route with ``max_tokens=2``: the model's bare-letter
answer could not be parsed as verbalised JSON and was hashed into noise,
which scored *below* chance on Swissmetro (J=3).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.outcomes.test_openai_client import _CapturingCompletions, _install_fake_openai


def _logprob_response(top: dict[str, float], text: str = "B") -> SimpleNamespace:
    entries = [SimpleNamespace(token=t, logprob=lp) for t, lp in top.items()]
    position = SimpleNamespace(token=text, logprob=max(top.values()), top_logprobs=entries)
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason="stop",
            logprobs=SimpleNamespace(content=[position]),
        )],
        model="served-model",
    )


def _plain_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                 finish_reason="stop", logprobs=None)],
        model="served-model",
    )


def test_openai_client_is_detected_and_logprobs_requested(monkeypatch) -> None:
    completions = _install_fake_openai(
        monkeypatch,
        _CapturingCompletions(response_factory=lambda: _logprob_response(
            {" B": -0.1, "A": -2.5, "(C": -3.0, "The": -4.0}
        )),
    )
    from src.baselines._llm_ranker_common import (
        _is_anthropic_client, _is_openai_client, call_llm_for_ranking,
    )
    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient(model_id="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                             base_url="http://127.0.0.1:8000/v1")
    assert _is_openai_client(client) and not _is_anthropic_client(client)

    probs = call_llm_for_ranking(client, "SYS", "USER", letters=("A", "B", "C"),
                                 max_tokens=2, seed=3)
    assert probs.shape == (3,) and abs(probs.sum() - 1.0) < 1e-9
    assert int(np.argmax(probs)) == 1  # " B" had the top logprob
    expected = np.exp([-2.5, -0.1, -3.0]); expected /= expected.sum()
    assert np.allclose(probs, expected)

    sent = completions.calls[-1]
    assert sent["logprobs"] is True and sent["top_logprobs"] == 20
    assert sent["max_tokens"] == 2 and sent["seed"] == 3
    assert sent["messages"][0]["content"].startswith("SYS")
    assert "single capital letter" in sent["messages"][0]["content"]
    assert sent["messages"][1] == {"role": "user", "content": "USER"}


def test_bare_letter_answer_is_parsed_when_no_logprobs(monkeypatch) -> None:
    _install_fake_openai(monkeypatch, _CapturingCompletions(
        response_factory=lambda: _plain_response("(C)")))
    from src.baselines._llm_ranker_common import call_llm_for_ranking
    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient(model_id="m", base_url="http://x/v1")
    probs = call_llm_for_ranking(client, "S", "U", letters=("A", "B", "C"))
    assert int(np.argmax(probs)) == 2 and probs[2] == pytest.approx(0.9)


def test_request_failure_falls_back_to_generate_path(monkeypatch) -> None:
    from tests.outcomes.test_openai_client import _FakeAPIConnectionError

    _install_fake_openai(monkeypatch, _CapturingCompletions(
        raise_exc=_FakeAPIConnectionError("down")))
    from src.baselines._llm_ranker_common import call_llm_for_ranking
    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient(model_id="m", base_url="http://x/v1")
    # Both the logprob call and generate() fail -> the generic path raises
    # the client's RuntimeError rather than silently returning noise.
    with pytest.raises(RuntimeError):
        call_llm_for_ranking(client, "S", "U", letters=("A", "B", "C"))


def test_extract_letter_logprobs_handles_openai_shapes() -> None:
    from src.baselines._llm_ranker_common import extract_letter_logprobs

    letters = ("A", "B", "C")
    lp = extract_letter_logprobs(_logprob_response({"A)": -0.2, "B": -1.0}), letters)
    assert int(np.argmax(lp)) == 0 and lp[2] == 0.0  # missing letter -> 0 after softmax
    bare = extract_letter_logprobs("b.", letters)
    assert int(np.argmax(bare)) == 1
    noise = extract_letter_logprobs("I would go with the second option", letters)
    assert noise.shape == (3,)  # hash fallback still well-defined
