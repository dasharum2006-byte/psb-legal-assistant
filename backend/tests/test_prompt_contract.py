"""Контракт вызова промпта и видимость ошибок.

Если build_rag_prompt перестанет принимать history (или поменяется иначе),
ошибка не должна выглядеть как «сервис недоступен» без следов в логе.
"""
from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.prompt import build_rag_prompt  # noqa: E402
from app.rag.orchestrator import ask  # noqa: E402


class FakeRetriever:
    def search(self, query, k=40, filters=None):
        return [{
            "doc_id": "doc", "doc": "ДОГОВОР ПОСТАВКИ № П-17/2024", "doc_type": "contract",
            "clause": "6.1", "text": "Приёмка по качеству 5 рабочих дней.",
            "score": 30.0, "status": "active",
        }]


class CapturingClient:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, system=None, **kwargs):
        self.prompt = prompt
        return "Приёмка 5 рабочих дней (п. 6.1)."


class BrokenClient:
    def generate(self, prompt, system=None, **kwargs):
        raise TypeError("build_rag_prompt() takes 2 positional arguments but 3 were given")


def test_prompt_accepts_question_chunks_history() -> None:
    params = list(inspect.signature(build_rag_prompt).parameters)
    assert params[:3] == ["question", "chunks", "history"]


def test_ask_passes_history_into_prompt() -> None:
    client = CapturingClient()
    history = [{"role": "user", "text": "какой гарантийный срок по договору с Альфа-Трейд?"}]

    ask("А за сколько дней проверить качество?", history, FakeRetriever(), client=client)

    assert "ИСТОРИЯ ДИАЛОГА" in client.prompt
    assert "Альфа-Трейд" in client.prompt


def test_type_error_is_logged_not_hidden(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="app.rag.orchestrator"):
        result = ask("вопрос", [], FakeRetriever(), client=BrokenClient())

    assert result["refusal"] is True              # пользователю — понятный отказ
    assert "PROMPT_SIGNATURE_ERROR" in caplog.text  # разработчику — след в логе
