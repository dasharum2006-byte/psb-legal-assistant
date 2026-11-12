#дароу
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.normalize import (  # noqa: E402
    normalize_clause,
    normalize_text,
    parse_amounts,
    tokens,
)


def test_normalize_text() -> None:
    assert normalize_text("Арендная  ПЛАТА «Северный»") == "арендная плата северный"
    assert normalize_text("Статья № 9") == "статья 9"
    assert normalize_text("Статья\u00a09") == "статья 9"
    assert normalize_text("  Ёлка   и ёж ") == "елка и еж"


def test_tokens() -> None:
    assert tokens("Арендная плата, 450 000 руб.") == [
        "арендная", "плата", "450", "000", "руб",
    ]
    assert tokens("Ёлка и ёж") == ["елка", "и", "еж"]
    assert tokens("А-08/2024") == ["а-08", "2024"]


def test_parse_amounts() -> None:
    assert parse_amounts("12 млн рублей") == [12000000]
    assert parse_amounts("3 000 000 рублей") == [3000000]
    assert parse_amounts("450 000 руб. в месяц") == [450000]
    assert parse_amounts("в августе 2024 года") == []
    assert parse_amounts("не более 7%") == []
    assert parse_amounts("0,1 % и 15 процентов") == []
    assert parse_amounts("2,5 млн, 45 тыс. и 2024 рубля") == [2500000, 45000, 2024]
    assert parse_amounts("1\u202f000\u2009000 ₽") == [1000000]


def test_normalize_clause() -> None:
    assert normalize_clause("п. 3.1") == "3.1"
    assert normalize_clause("(п. 3.1).") == "3.1"
    assert normalize_clause("пункт 3.1.2") == "3.1.2"
    assert normalize_clause("Источник: ТК РФ, ст. 70.") == "70"
    assert normalize_clause("ст. 9, ч. 2") == "9.2"
    assert normalize_clause("ч. 2") == "2"
    assert normalize_clause("п. 1") == "1"
    assert normalize_clause("ст. 13.11") == "13.11"
    assert normalize_clause("никаких ссылок тут нет") is None
