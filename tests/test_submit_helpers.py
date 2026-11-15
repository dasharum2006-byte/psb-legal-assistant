"""Регрессии критичных helper'ов зачётного solution.py.

solution.py уходит на платформу как есть: мелкая правка перед дедлайном
может сломать формат ссылки, проверку выдуманных сумм или выбор предложения
с суммой. Эти тесты фиксируют текущие контракты без обращения к сети.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import submit.solution as solution  # noqa: E402


def test_has_amount_detects_values_but_not_dates_or_clauses() -> None:
    assert solution.has_amount("Арендная плата — 450 000 рублей в месяц.")
    assert solution.has_amount("Ставка 0,1% годовых.")
    assert solution.has_amount("Штраф 500 рублей.")
    assert solution.has_amount("в течение 10 дней")
    assert solution.has_amount("50%")

    # дата и номер пункта — не «величина» для выбора предложения
    assert not solution.has_amount("не позднее 5-го числа")
    assert not solution.has_amount("пункт 3.1")
    assert not solution.has_amount("Арендодатель передаёт помещение.")


CHUNKS = [
    {
        "doc": "ДОГОВОР АРЕНДЫ № А-08/2024",
        "clause": "3.1",
        "text": "Арендная плата составляет 450 000 рублей в месяц, включая НДС.",
    },
    {
        "doc": "ТК РФ",
        "clause": "ст. 70",
        "text": "Срок испытания не может превышать трёх месяцев.",
    },
    {
        "doc": "Служебная записка № ЮО-7",
        "clause": "абз. 1",
        "text": "Оригиналы выдаются в течение 5 рабочих дней.",
    },
]


def test_grounded_accepts_only_citations_and_numbers_from_context() -> None:
    assert solution._grounded("Плата — 450 000 рублей (п. 3.1).", CHUNKS)
    assert solution._grounded("Плата — 450000 рублей (п. 3.1).", CHUNKS)
    assert not solution._grounded("Плата — 500 000 рублей (п. 3.1).", CHUNKS)
    assert not solution._grounded("Плата — 450 000 рублей (п. 9.9).", CHUNKS)

    assert solution._grounded("Испытание — не более трёх месяцев (ст. 70).", CHUNKS)
    assert solution._grounded("Испытание — не более трёх месяцев (статье 70).", CHUNKS)
    assert not solution._grounded("Испытание — не более трёх месяцев (ст. 99).", CHUNKS)
    assert not solution._grounded("Испытание — не более трёх месяцев (статье 99).", CHUNKS)
    assert solution._grounded("Оригиналы выдаются в течение 5 рабочих дней (абз. 1).", CHUNKS)
    assert not solution._grounded("Оригиналы выдаются в течение 5 рабочих дней (абз. 2).", CHUNKS)

    # «3.1.1» подтверждается родительским пунктом «3.1» из нарезки
    assert solution._grounded("Плата — 450 000 рублей (п. 3.1.1).", CHUNKS)


def test_ensure_citation_keeps_existing_and_adds_missing() -> None:
    text = "Плата 450 000 рублей"
    assert solution._ensure_citation(f"{text} (п. 3.1).", CHUNKS) == f"{text} (п. 3.1)."
    assert solution._ensure_citation(f"{text} (п.3.1).", CHUNKS) == f"{text} (п.3.1)."
    assert solution._ensure_citation("Согласно статье 70 ТК РФ.", CHUNKS) == "Согласно статье 70 ТК РФ."
    assert solution._ensure_citation("Выдаются по заявке (абз. 1).", CHUNKS) == "Выдаются по заявке (абз. 1)."
    assert solution._ensure_citation("Выдаются по заявке (абзаце 1).", CHUNKS) == "Выдаются по заявке (абзаце 1)."
    assert solution._ensure_citation("Выдаются по заявке (№ ЮО-7).", CHUNKS) == "Выдаются по заявке (№ ЮО-7)."

    assert solution._ensure_citation(f"{text}.", CHUNKS).endswith("(п. 3.1)")
    assert solution._ensure_citation("Испытание три месяца.", [CHUNKS[1]]).endswith("(ст. 70)")
    assert solution._ensure_citation("Выдаются по заявке.", [CHUNKS[2]]).endswith("(абз. 1)")


def test_grounded_ignores_kopecks_and_non_breaking_spaces() -> None:
    assert solution._grounded("Плата — 450 000,00 рублей (п. 3.1).", CHUNKS)
    assert solution._grounded("Плата — 450 000.00 рублей (п. 3.1).", CHUNKS)
    assert solution._grounded("Плата — 450\u00a0000 рублей (п. 3.1).", CHUNKS)
    # другая сумма по-прежнему не подтверждается
    assert not solution._grounded("Плата — 45 000 000,00 рублей (п. 3.1).", CHUNKS)

    # копейки могут быть в самом документе, а в ответе — без них
    with_kopecks = [{
        "doc": "ДОГОВОР АРЕНДЫ № А-08/2024",
        "clause": "3.1",
        "text": "Арендная плата составляет 450 000,00 рублей в месяц.",
    }]
    assert solution._grounded("Плата — 450 000 рублей (п. 3.1).", with_kopecks)


def test_pick_sentences_prefers_value_when_asked() -> None:
    text = (
        "Арендная плата вносится ежемесячно согласно условиям договора. "
        "Ежемесячный размер составляет 450 000 рублей. "
        "Стороны подписали акт приёма-передачи без замечаний."
    )
    chunk = {"doc": "ДОГОВОР", "clause": "3.1", "text": text}
    wanted = solution.significant_tokens("какая арендная плата")

    by_value = solution._pick_sentences(chunk, wanted, want_value=True)
    without_value = solution._pick_sentences(chunk, wanted, want_value=False)

    assert "450 000" in by_value
    assert by_value.startswith("Ежемесячный размер составляет")
    assert "450 000" not in without_value
    assert without_value.startswith("Арендная плата вносится")


def test_pick_sentences_skips_blacklisted_and_handles_short_text() -> None:
    chunk = {
        "doc": "ДОГОВОР",
        "clause": "8.1",
        "text": (
            "Реквизиты сторон: ИНН 7701234567, КПП 770101001, расчётный счёт в банке. "
            "Оригиналы договоров выдаются в течение пяти рабочих дней по заявке."
        ),
    }
    wanted = solution.significant_tokens("какие реквизиты и ИНН указаны в документе")
    picked = solution._pick_sentences(chunk, wanted)
    assert "ИНН" not in picked
    assert "Оригиналы договоров" in picked

    short = {"doc": "ПИСЬМО", "clause": "абз. 1", "text": "Короткий пункт. Ещё короче."}
    assert solution._pick_sentences(short, set()) == "Короткий пункт. Ещё короче."


def test_grounded_compares_numbers_not_substrings() -> None:
    big = [{
        "doc": "ДОГОВОР",
        "clause": "3.1",
        "text": "Цена договора — 4 500 000 рублей.",
    }]
    assert not solution._grounded("Цена — 450 000 рублей (п. 3.1).", big)
    assert solution._grounded("Цена — 4 500 000 рублей (п. 3.1).", big)

    small = [{
        "doc": "ДОГОВОР",
        "clause": "3.1",
        "text": "Штраф — 300 рублей за день просрочки.",
    }]
    assert not solution._grounded("Штраф — 500 рублей (п. 3.1).", small)
    assert solution._grounded("Штраф — 300 рублей (п. 3.1).", small)


def test_grounded_checks_percentages_and_article_context() -> None:
    chunks = [{
        "doc": "ДОГОВОР",
        "clause": "3.1",
        "text": "Ставка 6% годовых. Срок — 70 дней.",
    }]
    assert not solution._grounded("Ставка 7% (п. 3.1).", chunks)
    assert solution._grounded("Ставка 6% (п. 3.1).", chunks)
    # «70 дней» в тексте — не основание подтверждать ссылку на ст. 70
    assert not solution._grounded("Согласно ст. 70 (ст. 70).", chunks)


def test_grounded_allows_numbers_from_the_question() -> None:
    chunks = [{
        "doc": "ДОГОВОР",
        "clause": "3.1",
        "text": "Арендная плата — 450 000 рублей, ставка 6% годовых.",
    }]
    # без вопроса «500 000» — выдумка
    assert not solution._grounded("Плата 500 000 рублей (п. 3.1).", chunks)

    # с вопросом модель может исправить неверную цифру из вопроса
    assert solution._grounded(
        "Плата составляет 450 000, а не 500 000 рублей (п. 3.1).",
        chunks,
        "Правда ли, что плата 500 000 рублей?",
    )
    assert solution._grounded(
        "Ставка 6%, а не 7% (п. 3.1).",
        chunks,
        "Правда ли, что ставка 7%?",
    )
    assert not solution._grounded(
        "Ставка 7% (п. 3.1).",
        chunks,
        "Правда ли, что ставка 5%?",
    )


def test_build_prompt_does_not_write_p_article_twice() -> None:
    chunks = [{
        "doc": "Статья 70. Испытание при приёме на работу",
        "clause": "ст. 70",
        "revision": "2025",
        "text": "Срок испытания не более трёх месяцев.",
    }]
    prompt = solution.build_prompt("какой срок испытания?", chunks)

    assert "п. ст. 70" not in prompt
    assert "ст. 70" in prompt
    assert "[ред. 2025]" in prompt


def test_grounded_short_terms_allows_word_numerals() -> None:
    chunks = [{
        "doc": "СЛУЖЕБНАЯ ЗАПИСКА № ЮО-7",
        "clause": "абз. 1",
        "text": "Оригиналы выдаются в течение трёх рабочих дней.",
    }]
    assert solution._grounded("Выдаются в течение 3 рабочих дня (абз. 1).", chunks)
    assert solution._grounded("Выдаются в течение трёх рабочих дней (абз. 1).", chunks)
    assert not solution._grounded("Выдаются в течение 30 рабочих дней (абз. 1).", chunks)


def test_grounded_short_terms_allow_numbers_from_question() -> None:
    chunks = [{
        "doc": "ДОГОВОР",
        "clause": "3.1",
        "text": "Срок оплаты — 30 календарных дней.",
    }]
    assert not solution._grounded("Срок оплаты — 10 дней (п. 3.1).", chunks)
    assert solution._grounded(
        "Срок оплаты — 10 дней (п. 3.1).",
        chunks,
        "Правда ли, что срок оплаты 10 дней?",
    )


def test_ensure_citation_recognizes_p_without_dot() -> None:
    chunks = [{"doc": "ДОГОВОР", "clause": "3.1", "text": ""}]
    text = "Плата 450 000 рублей (п 3.1)."

    assert solution._ensure_citation(text, chunks) == text
