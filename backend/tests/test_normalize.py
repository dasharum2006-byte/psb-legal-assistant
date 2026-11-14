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


def test_amounts_with_dot_as_thousands_separator() -> None:
    """«1.500.000 руб.» — это полтора миллиона, а не 1 рубль и не пусто."""
    assert parse_amounts("1.500.000 руб.") == [1_500_000]
    assert parse_amounts("1.234.567 рублей") == [1_234_567]


def test_amounts_with_spaces() -> None:
    assert parse_amounts("450 000 рублей") == [450_000]
    assert parse_amounts("12 000 000 рублей") == [12_000_000]


def test_fractional_millions_and_thousands() -> None:
    assert parse_amounts("1,5 млн") == [1_500_000]
    assert parse_amounts("1,5 тыс.") == [1_500]


def test_kopeks_are_dropped_not_multiplied() -> None:
    assert parse_amounts("100,05 руб.") == [100]


def test_percents_years_and_articles_are_not_amounts() -> None:
    assert parse_amounts("не более 7%") == []
    assert parse_amounts("0,1% от суммы") == []
    assert parse_amounts("в августе 2024 года") == []
    assert parse_amounts("ст. 13.11 КоАП") == []


def test_billions_are_parsed() -> None:
    """«2 млрд» — это два миллиарда, а не 2 рубля."""
    assert parse_amounts("2 млрд") == [2_000_000_000]
    assert parse_amounts("3 миллиарда рублей") == [3_000_000_000]
    assert parse_amounts("1,5 млрд") == [1_500_000_000]
    assert parse_amounts("2 миллиардами рублей") == [2_000_000_000]


def test_abbreviated_thousands() -> None:
    """«т.р.» и «т. руб.» — тоже тысячи."""
    assert parse_amounts("100 т.р.") == [100_000]
    assert parse_amounts("100 т. руб.") == [100_000]
    assert parse_amounts("100 т.руб.") == [100_000]
    assert parse_amounts("100т.р.") == [100_000]


def test_thousands_words() -> None:
    assert parse_amounts("500 тысяч рублей") == [500_000]
    assert parse_amounts("500 тыс. руб.") == [500_000]
    assert parse_amounts("250 тыс") == [250_000]
