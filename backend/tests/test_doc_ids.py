"""Опознание документов в вопросе: границы слов и похожие названия."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.orchestrator import find_doc_ids  # noqa: E402


def test_counterparties_are_not_confused() -> None:
    assert find_doc_ids("гарантия по договору с «Альфа-Трейдинг»") == ["ПТ-05/2023"]
    assert find_doc_ids("гарантия по договору с «Альфа-Трейд»") == ["П-17/2024"]


def test_document_numbers_respect_word_boundaries() -> None:
    assert find_doc_ids("договор К-123") == []
    assert find_doc_ids("письмо К-12") == ["К-12"]


def test_number_and_name_do_not_duplicate() -> None:
    assert find_doc_ids("аренда у «Северного Склада», договор А-08/2024") == ["А-08/2024"]


def test_multiple_documents_in_one_question() -> None:
    assert find_doc_ids("договор А-08/2024 и П-17/2024") == ["А-08/2024", "П-17/2024"]


def test_no_documents_found() -> None:
    assert find_doc_ids("какая арендная плата?") == []
