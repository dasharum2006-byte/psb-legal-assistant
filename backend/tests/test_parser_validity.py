"""Статус редакции: «утратила силу» важнее наличия даты начала действия."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.parser import _validity  # noqa: E402


def test_both_phrases_means_repealed() -> None:
    """В шапке есть и «действует с», и «утратила силу» — документ отменён."""
    text = (
        "РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ\n"
        "Редакция действует с 01.01.2023. "
        "Редакция утратила силу с 01.01.2025 в связи с новой редакцией.\n"
    )
    valid_from, valid_to, status = _validity(text)
    assert valid_from == "2023-01-01"
    assert valid_to == "2025-01-01"
    assert status == "repealed"


def test_only_active_since() -> None:
    valid_from, valid_to, status = _validity("Действует с 01.03.2025. Редакция от 17.02.2025.")
    assert (valid_from, valid_to, status) == ("2025-03-01", None, "active")


def test_only_repealed() -> None:
    valid_from, valid_to, status = _validity("РЕДАКЦИЯ УТРАТИЛА СИЛУ с 01.03.2025")
    assert (valid_from, valid_to, status) == (None, "2025-03-01", "repealed")


def test_no_dates_is_active() -> None:
    assert _validity("ДОГОВОР ПОСТАВКИ № 1") == (None, None, "active")
