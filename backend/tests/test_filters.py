"""Фильтры поиска из запроса API: doc_type, status, doc_ids, as_of.

Проверяем, что фильтры доходят до `ask()` и реально отсекают документы.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.orchestrator import ask  # noqa: E402


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k=8, filters=None):
        return [dict(c) for c in self.chunks]


class FakeClient:
    """Цитирует первый пункт, который увидел в промпте, — как настоящая модель."""

    def generate(self, prompt, system=None, **kwargs):
        import re
        # шапка промпта: «Источник: <док>, раздел «…», пункт: 3.1» или «пункт/статья: ст. 70»
        clause = re.search(r"(?:пункт/статья|пункт|статья):\s*([^,\n]+)", prompt)
        if not clause:
            return "Не нашёл."
        value = clause.group(1).strip()
        citation = f"({value})" if value.startswith("ст.") else f"(п. {value})"
        return f"Согласно документам, условие действует {citation}."


CHUNKS = [
    {"doc_id": "dogovor_a-08", "doc": "ДОГОВОР АРЕНДЫ № А-08/2024", "doc_type": "contract",
     "clause": "1.1", "text": "Арендная плата 450 000 рублей.", "score": 30.0,
     "status": "active", "valid_from": "2024-08-01", "valid_to": None},
    {"doc_id": "reglament_2023", "doc": "РЕГЛАМЕНТ (ред. 2023)", "doc_type": "regulation",
     "clause": "1.1", "text": "Порог одобрения 10 000 000 рублей.", "score": 28.0,
     "status": "repealed", "valid_from": None, "valid_to": "2025-03-01"},
    {"doc_id": "tk_st70", "doc": "Статья 70. Испытание при приёме на работу", "doc_type": "law",
     "clause": "ст. 70", "text": "Срок испытания не более трёх месяцев.", "score": 25.0,
     "status": "active", "valid_from": None, "valid_to": None},
]


def run(question: str, filters: dict) -> dict:
    return ask(question, [], FakeRetriever(CHUNKS), client=FakeClient(), filters=filters)


def docs_in(result: dict) -> set[str]:
    return {source["doc"] for source in result["sources"]}


def test_filter_by_doc_type_keeps_only_law() -> None:
    result = run("какой срок испытания?", {"doc_type": ["law"]})
    assert result["refusal"] is False
    assert docs_in(result) == {"Статья 70. Испытание при приёме на работу"}


def test_filter_by_status_keeps_only_active() -> None:
    result = run("какой порог одобрения?", {"status": ["active"]})
    assert "РЕГЛАМЕНТ (ред. 2023)" not in docs_in(result)


def test_filter_by_doc_ids_keeps_single_document() -> None:
    result = run("какая арендная плата?", {"doc_ids": ["а-08/2024"]})
    assert docs_in(result) == {"ДОГОВОР АРЕНДЫ № А-08/2024"}


def test_filter_by_as_of_drops_future_document() -> None:
    result = run("что действовало в 2024 году?", {"as_of": "2024-08-31"})
    assert "РЕГЛАМЕНТ (ред. 2023)" in docs_in(result)

    after = run("что действует сейчас?", {"as_of": "2026-01-01"})
    assert "РЕГЛАМЕНТ (ред. 2023)" not in docs_in(after)


def test_no_filters_keeps_everything() -> None:
    result = run("общий вопрос про документы", {})
    assert result["refusal"] is False
    assert len(docs_in(result)) >= 1


class DeepRetriever:
    """Правильный документ лежит глубоко в выдаче — на 15-м месте."""

    def __init__(self, total: int = 30, target_at: int = 14):
        self.total = total
        self.target_at = target_at

    def search(self, query, k=8, filters=None):
        chunks = []
        for position in range(self.total):
            is_target = position == self.target_at
            chunks.append({
                "doc_id": "target_doc" if is_target else f"noise_{position}",
                "doc": "РЕГЛАМЕНТ (ред. 2025)" if is_target else f"Шум {position}",
                "doc_type": "regulation" if is_target else "contract",
                "clause": "1.3" if is_target else "1.1",
                "text": "Инициатором согласования является руководитель подразделения." if is_target
                        else "Посторонний текст про аренду и поставку.",
                "score": 30.0 - position,
                "status": "active",
            })
        return chunks[:k]


def test_filter_finds_document_beyond_first_eight() -> None:
    """Раньше поиск брал 8 кандидатов, фильтр их выкидывал и был отказ."""
    result = ask("кто запускает согласование?", [], DeepRetriever(), client=FakeClient(),
                 filters={"doc_type": ["regulation"]})
    assert result["refusal"] is False, "верный документ был на 15-м месте и не нашёлся"
    assert result["sources"][0]["doc"] == "РЕГЛАМЕНТ (ред. 2025)"


class CosineRetriever:
    """Гибрид/эмбеддинги: очки в шкале 0..1 (как косинус или RRF)."""

    def __init__(self, scores):
        self.scores = scores

    def search(self, query, k=40, filters=None):
        return [{
            "doc_id": "doc", "doc": "ДОГОВОР АРЕНДЫ № А-08/2024", "doc_type": "contract",
            "clause": "3.1", "text": "Арендная плата 450 000 рублей.",
            "score": score, "status": "active",
        } for score in self.scores]


def test_cosine_scale_scores_are_not_rejected() -> None:
    """Порог 8 на шкале 0..1 выбрасывал бы все чанки и давал вечный отказ."""
    result = ask("какая арендная плата?", [], CosineRetriever([0.82, 0.71, 0.55]),
                 client=FakeClient(), filters={})
    assert result["refusal"] is False, "очки 0..1 не должны отсекаться порогом 8"


def test_low_cosine_scores_still_refuse() -> None:
    """Если по смыслу ничего не подошло — отказ остаётся отказом."""
    result = ask("какой размер пособия?", [], CosineRetriever([0.10, 0.08]),
                 client=FakeClient(), filters={})
    assert result["refusal"] is True


def test_bm25_scale_still_uses_absolute_threshold() -> None:
    result = ask("вопрос", [], CosineRetriever([40.0, 22.0]), client=FakeClient(), filters={})
    assert result["refusal"] is False
