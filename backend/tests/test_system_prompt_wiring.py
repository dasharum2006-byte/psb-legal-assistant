"""SYSTEM_PROMPT обязан доходить до модели.

Раньше он лежал в prompt.py, но в API не передавался: оркестратор звал
client.generate(prompt) без system, и модель не видела ни формата ссылок,
ни защиты от инъекций, ни примеров.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.client import LLMClient  # noqa: E402
from app.llm.prompt import SYSTEM_PROMPT, build_rag_prompt  # noqa: E402
from app.rag.orchestrator import ask  # noqa: E402

CHUNKS = [{"doc_id": "tk", "doc": "ТК РФ", "doc_type": "law", "clause": "ст. 70",
           "text": "Срок испытания не более трёх месяцев.", "score": 30.0, "status": "active"}]


class Spy:
    def __init__(self) -> None:
        self.system: str | None = None
        self.prompt: str | None = None

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        self.system, self.prompt = system, prompt
        return "Срок испытания — три месяца (ст. 70)."


class Retriever:
    def search(self, query: str, k: int = 40, filters=None):
        return [dict(CHUNKS[0])]


def test_ask_passes_system_prompt() -> None:
    spy = Spy()
    ask("какой срок испытания?", [], Retriever(), client=spy)
    assert spy.system is SYSTEM_PROMPT
    assert "ИНЪЕКЦ" in spy.system


def test_context_goes_to_user_message() -> None:
    spy = Spy()
    ask("какой срок испытания?", [], Retriever(), client=spy)
    assert "Срок испытания не более трёх месяцев" in spy.prompt
    assert "ВОПРОС ПОЛЬЗОВАТЕЛЯ" in spy.prompt


def test_prompt_builder_keeps_rules_out_of_user_message() -> None:
    """Правила живут в system, а не дублируются в пользовательском тексте."""
    prompt = build_rag_prompt("вопрос", CHUNKS)
    assert "ЗАЩИТА ОТ ИНЪЕКЦИЙ" not in prompt


def test_client_sends_system_message_to_api() -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "ответ"}, "finish_reason": "stop"}]}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, headers=None, json=None):
            captured.update(json or {})
            return FakeResponse()

    with patch.dict("os.environ", {"LLM_API_KEY": "test", "LLM_BASE_URL": "https://example.test/v1"}, clear=False):
        with patch("app.llm.client._make_client", lambda timeout: FakeClient()):
            client = LLMClient()
            client.generate("вопрос", system="ПРАВИЛА")

    messages = captured["messages"]
    assert messages[0] == {"role": "system", "content": "ПРАВИЛА"}
    assert messages[1]["role"] == "user"
