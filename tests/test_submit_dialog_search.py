"""Поиск в solution.py должен учитывать историю диалога.

Раньше work() собирал search_text = история + последняя реплика, но
answer_question() получал только последнюю реплику. На уточняющих вопросах
(«А за сколько дней после поставки?») BM25 и семантика искали без названия
договора из предыдущей реплики.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import submit.solution as solution  # noqa: E402


class RecordingRetriever:
    """Ничего не находит, но запоминает, с каким запросом его позвали."""

    chunks: list[dict] = []

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.topic_queries: list[str] = []

    def search(self, query: str, k: int = solution.TOP_K, topic_query: str = "") -> list[dict]:
        self.queries.append(query)
        self.topic_queries.append(topic_query)
        return []


HISTORY = (
    "пользователь: какой срок поставки по договору с ООО «Альфа-Трейд»? "
    "ассистент: поставка — 10 рабочих дней."
)
LAST_QUESTION = "А за сколько дней после поставки?"


def test_answer_question_uses_passed_dialog_search_query(monkeypatch) -> None:
    monkeypatch.setattr(solution, "api_available", lambda: False)
    retriever = RecordingRetriever()

    result = solution.answer_question(
        None, LAST_QUESTION, retriever, HISTORY, f"{HISTORY} {LAST_QUESTION}".strip()
    )

    assert result == solution.REFUSAL
    assert retriever.queries, "поиск не вызывался"
    assert "альфа-трейд" in retriever.queries[0].lower()
    assert LAST_QUESTION.lower() in retriever.queries[0].lower()
    assert retriever.topic_queries[0] == LAST_QUESTION


def test_answer_question_builds_dialog_query_if_not_passed(monkeypatch) -> None:
    monkeypatch.setattr(solution, "api_available", lambda: False)
    retriever = RecordingRetriever()

    solution.answer_question(None, LAST_QUESTION, retriever, HISTORY)

    assert retriever.queries
    assert "альфа-трейд" in retriever.queries[0].lower(), (
        "без явного search_query функция должна сама склеить историю и вопрос"
    )


def test_extractive_answer_uses_dialog_tokens() -> None:
    text = (
        "Гарантийный срок на товар составляет двадцать четыре месяца с даты поставки "
        "и действует при условии соблюдения всех правил эксплуатации и хранения. "
        "Гарантийный срок продлевается на время ремонта и не включает периоды "
        "нарушения покупателем условий эксплуатации, если это установлено договором. "
        "Приёмка товара по качеству производится в течение 5 рабочих дней после поставки."
    )
    chunk = {
        "doc": "ДОГОВОР ПОСТАВКИ № П-17/2024",
        "clause": "6.1",
        "text": text,
        "score": 20.0,
    }
    last_question = "А за сколько дней?"
    search_query = "за сколько дней проверить качество по договору с Альфа-Трейд?"

    answer = solution.extractive_answer(last_question, [chunk], search_query)

    assert "5 рабочих дней" in answer
    assert answer.startswith("Приёмка товара по качеству")

