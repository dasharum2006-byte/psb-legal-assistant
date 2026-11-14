"""Диалоговые вопросы: история из API должна попадать в поиск.

История приходит в формате {"role","content"} (OpenAI-стиль), а код читал
только {"role","text"} — уточняющие вопросы искали не то.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.orchestrator import ask, contextual_query  # noqa: E402

CONTENT_HISTORY = [
    {"role": "user", "content": "На какой срок выдают оригиналы договоров из архива?"},
    {"role": "assistant", "content": "На срок не более 3 рабочих дней (Служебная записка № ЮО-7)."},
]
TEXT_HISTORY = [{"role": "user", "text": "На какой срок выдают оригиналы договоров из архива?"}]


def test_content_style_history_is_used() -> None:
    merged = contextual_query("А если заявку подал не руководитель?", CONTENT_HISTORY)
    assert "оригиналы договоров" in merged.lower()
    assert merged.endswith("А если заявку подал не руководитель?")


def test_text_style_history_still_works() -> None:
    merged = contextual_query("А если заявку подал не руководитель?", TEXT_HISTORY)
    assert "оригиналы договоров" in merged.lower()


def test_long_question_is_not_merged() -> None:
    """Самодостаточный вопрос не надо склеивать с предыдущим."""
    question = " ".join(["слово"] * 15)
    assert contextual_query(question, CONTENT_HISTORY) == question


def test_history_reaches_llm_prompt() -> None:
    seen: dict = {}

    class Spy:
        def generate(self, prompt, system=None, **kwargs):
            seen["prompt"] = prompt
            return "На срок не более 3 рабочих дней (Служебная записка № ЮО-7)."

    class Retriever:
        def search(self, query, k=40, filters=None):
            return [{"doc_id": "юо-7", "doc": "СЛУЖЕБНАЯ ЗАПИСКА № ЮО-7 от 12.05.2025",
                     "clause": "абз. 1", "number": "ЮО-7", "doc_type": "memo",
                     "text": "Оригиналы договоров выдаются по заявке на срок не более 3 рабочих дней.",
                     "score": 20.0, "status": "active"}]

    ask("А если заявку подал не руководитель?", CONTENT_HISTORY, Retriever(), client=Spy())
    assert "ИСТОРИЯ ДИАЛОГА" in seen["prompt"]
    assert "оригиналы договоров" in seen["prompt"].lower()
