"""Сильный BM25-результат не должен превращаться в отказ.

Раньше best_word_score считался только по score <= 60. Если все кандидаты
были сильнее, максимум получался 0.0, и при недоступной семантике решение
отказывалось отвечать даже при точном попадании.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import submit.solution as solution  # noqa: E402


class HighScoreRetriever:
    chunks: list[dict] = []

    def search(self, query: str, k: int = solution.TOP_K, topic_query: str = "") -> list[dict]:
        return [{
            "doc": "ДОГОВОР АРЕНДЫ № А-08/2024",
            "number": "А-08/2024",
            "clause": "3.1",
            "text": "Арендная плата составляет 450 000 рублей в месяц.",
            "searchable": (
                "ДОГОВОР АРЕНДЫ № А-08/2024 3.1 "
                "Арендная плата составляет 450 000 рублей в месяц."
            ),
            "score": 100.0,
        }]


def test_strong_bm25_is_not_refused_when_semantic_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(solution, "api_available", lambda: False)

    answer = solution.answer_question(
        None, "Какая арендная плата по договору с Северным Складом?", HighScoreRetriever()
    )

    assert answer != solution.REFUSAL
    assert "450 000" in answer
    assert "(п. 3.1)" in answer
