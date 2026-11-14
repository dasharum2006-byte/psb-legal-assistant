"""as_of включается только при временном смысле вопроса.

Год в номере договора или дате документа («договор от 15 марта 2023 года»)
не означает вопрос о состоянии на дату. Иначе фильтр as_of отсекал бы
актуальные документы.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.orchestrator import analyze_query, has_temporal_intent  # noqa: E402


def test_date_of_document_is_not_as_of() -> None:
    assert has_temporal_intent("Какие условия расторжения в договоре от 15 марта 2023 года?") is False
    assert analyze_query("Какие условия расторжения в договоре от 15 марта 2023 года?")["as_of"] is None


def test_letter_date_is_not_as_of() -> None:
    assert analyze_query("Письмо комплаенс от 15.04.2025 — что в нём?")["as_of"] is None


def test_contract_number_year_is_not_as_of() -> None:
    assert analyze_query("Какая арендная плата по договору А-08/2024?")["as_of"] is None


def test_state_on_date_is_as_of() -> None:
    assert analyze_query("Что действовало на 15.04.2025?")["as_of"] == "2025-04-15"
    assert analyze_query("Правила по состоянию на 1 августа 2024")["as_of"] == "2024-08-01"


def test_period_questions_are_as_of() -> None:
    # вопрос 5 кейса: важно, какая редакция действовала в августе 2024
    assert analyze_query(
        "Договор на 12 млн рублей заключали в августе 2024 года. Нужно ли было одобрение?"
    )["as_of"] == "2024-08-31"
    assert analyze_query("Какие правила были в 2023 году?")["as_of"] == "2023-12-31"
