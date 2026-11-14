"""Источники отбираются по каноническим ключам ссылок, а не по подстроке.

«статья 15», «ст.15» и «ст. 15» — одна и та же ссылка. Раньше строгое
сравнение не находило её, и в источники попадало всё подряд.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.orchestrator import pick_sources  # noqa: E402

CHUNKS = [
    {"doc": "ДОГОВОР АРЕНДЫ № А-08/2024", "clause": "3.1", "text": "плата", "score": 30.0},
    {"doc": "ДОГОВОР АРЕНДЫ № А-08/2024", "clause": "3.2", "text": "срок", "score": 28.0},
    {"doc": "ТК РФ", "clause": "ст. 15", "text": "норма", "score": 26.0},
]


def clauses(result: list[dict]) -> list[str]:
    return [source["clause"] for source in result]


def test_law_citation_in_any_format_is_found() -> None:
    for answer in ("согласно статье 15", "по ст.15", "по ст. 15 ТК РФ"):
        assert clauses(pick_sources(answer, CHUNKS)) == ["ст. 15"], answer


def test_point_citation_without_space_is_found() -> None:
    assert clauses(pick_sources("Плата 450 000 (п.3.1).", CHUNKS)) == ["3.1"]


def test_all_cited_clauses_are_returned() -> None:
    result = pick_sources("Плата (п. 3.1), срок (п. 3.2).", CHUNKS)
    assert clauses(result) == ["3.1", "3.2"]


def test_answer_without_citations_falls_back_to_top() -> None:
    result = pick_sources("ответ без ссылок", CHUNKS)
    assert len(result) == 3


def test_invented_citation_does_not_pick_wrong_source() -> None:
    result = pick_sources("Плата (п. 9.9).", CHUNKS)
    # ссылки нет в чанках -> отдаём топ, но не приписываем 9.9 источнику
    assert "9.9" not in clauses(result)
