"""Метаданные в поиске solution.py не должны теряться.

Бэкенд уже умеет бустить чанки по `number` и `clause`, а в сабмите номер
искался только в заголовке и только после знака «№». Из-за этого вопросы
«Что сказано в записке ЮО-7?» и «что в п. 3.2?» не получали буста.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import submit.solution as solution  # noqa: E402


def test_document_number_is_extracted_without_number_sign() -> None:
    assert "юо-7" in solution.question_doc_numbers("Что сказано в записке ЮО-7?")
    assert "а-08" in solution.question_doc_numbers("По договору А-08/2024 про плату")
    assert solution.question_doc_numbers("просто вопрос без номера") == set()


def test_metadata_boost_uses_chunk_number() -> None:
    memo = {"number": "ЮО-7", "doc": "СЛУЖЕБНАЯ ЗАПИСКА", "clause": "абз. 1"}
    other = {"number": "К-12", "doc": "ИСХОДЯЩЕЕ ПИСЬМО", "clause": "абз. 1"}

    assert solution.metadata_boost("что в записке ЮО-7", memo) >= 15.0
    assert solution.metadata_boost("что в записке ЮО-7", other) == 0.0


def test_metadata_boost_rewards_explicit_clause() -> None:
    chunk = {"number": "А-08/2024", "doc": "ДОГОВОР АРЕНДЫ", "clause": "3.1"}
    assert solution.metadata_boost("что написано в п. 3.1", chunk) >= 10.0
    assert solution.metadata_boost("что написано в договоре", chunk) == 0.0


def test_search_finds_memo_by_number_without_number_sign() -> None:
    memo = {
        "doc": "СЛУЖЕБНАЯ ЗАПИСКА № ЮО-7",
        "number": "ЮО-7",
        "clause": "абз. 1",
        "text": "Оригиналы выдаются по заявке.",
        "searchable": "СЛУЖЕБНАЯ ЗАПИСКА № ЮО-7 абз. 1 Оригиналы выдаются по заявке.",
    }
    letter = {
        "doc": "ИСХОДЯЩЕЕ ПИСЬМО № К-12",
        "number": "К-12",
        "clause": "абз. 1",
        "text": "Оригиналы выдаются по заявке.",
        "searchable": "ИСХОДЯЩЕЕ ПИСЬМО № К-12 абз. 1 Оригиналы выдаются по заявке.",
    }
    retriever = solution.Retriever([memo, letter])

    result = retriever.search("Что сказано в записке ЮО-7?", k=2)

    assert result
    assert result[0]["number"] == "ЮО-7"


def test_search_prioritizes_explicit_clause() -> None:
    chunks = [
        {
            "doc": "ДОГОВОР АРЕНДЫ № А-08/2024",
            "number": "А-08/2024",
            "clause": clause,
            "text": "Арендная плата вносится ежемесячно.",
            "searchable": (
                f"ДОГОВОР АРЕНДЫ № А-08/2024 {clause} "
                "Арендная плата вносится ежемесячно."
            ),
        }
        for clause in ("3.1", "3.2")
    ]
    # разбавляем корпус, чтобы IDF по общим словам был положительным
    for index in range(5):
        chunks.append({
            "doc": f"ПИСЬМО № К-{index}",
            "number": f"К-{index}",
            "clause": "абз. 1",
            "text": f"Поставка товара по заявке номер {index}.",
            "searchable": f"ПИСЬМО № К-{index} абз. 1 Поставка товара по заявке номер {index}.",
        })
    retriever = solution.Retriever(chunks)

    result = retriever.search("что написано в п. 3.2 про арендную плату?", k=2)

    assert result
    assert result[0]["clause"] == "3.2"


def test_quoted_names_are_distinguished_exactly() -> None:
    trade = {"doc": "ДОГОВОР ПОСТАВКИ с ООО «Альфа-Трейд»", "text": "Поставка товара."}
    trading = {"doc": "ДОГОВОР ПОСТАВКИ с ООО «Альфа-Трейдинг»", "text": "Поставка товара."}

    assert solution.question_names("что с «Альфа-Трейд»?") == {"альфа-трейд"}
    assert solution.exact_mention_boost("что с «Альфа-Трейд»?", trade) == 25.0
    assert solution.exact_mention_boost("что с «Альфа-Трейд»?", trading) == 0.0


def test_repealed_penalty_depends_on_past_year_in_question() -> None:
    old = {"repealed": True}
    assert solution.repealed_penalty("что действует сейчас?", old) == -12.0
    assert solution.repealed_penalty("что действовало в 2023 году?", old) == 0.0

    # год в номере договора — не вопрос про прошлую редакцию
    assert solution.has_past_year("по договору А-08/2024") is False
    assert solution.has_past_year("на 2023 год") is True


def test_search_prefers_active_edition() -> None:
    chunks = [
        {
            "doc": "РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ",
            "number": "Р-1",
            "clause": "1.1",
            "text": "Согласование занимает пять рабочих дней.",
            "searchable": "РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ 1.1 Согласование занимает пять рабочих дней.",
            "status": "active",
        },
        {
            "doc": "РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ",
            "number": "Р-1",
            "clause": "1.1",
            "text": "Согласование занимает пять рабочих дней.",
            "searchable": "РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ 1.1 Согласование занимает пять рабочих дней.",
            "repealed": True,
        },
    ]
    for index in range(5):
        chunks.append({
            "doc": f"ПИСЬМО № К-{index}",
            "number": f"К-{index}",
            "clause": "абз. 1",
            "text": f"Поставка товара по заявке номер {index}.",
            "searchable": f"ПИСЬМО № К-{index} абз. 1 Поставка товара по заявке номер {index}.",
            "status": "active",
        })
    retriever = solution.Retriever(chunks)

    result = retriever.search("сколько дней занимает согласование договора?", k=2)

    assert result
    assert result[0].get("repealed") is not True
    assert result[0].get("status") == "active"


def test_foreign_company_gets_small_penalty() -> None:
    trade = {"doc": "ДОГОВОР ПОСТАВКИ с ООО «Альфа-Трейд»", "text": "Поставка товара."}
    trading = {"doc": "ДОГОВОР ПОСТАВКИ с ООО «Альфа-Трейдинг»", "text": "Поставка товара."}
    query = "что с «Альфа-Трейд»?"

    names = solution.relevant_quoted_names(query, [trade, trading])
    assert names == {"альфа-трейд"}
    assert solution.foreign_name_penalty(query, trade, names) == 0.0
    assert solution.foreign_name_penalty(query, trading, names) == solution.FOREIGN_NAME_PENALTY

    # «Меридиан» — своя компания, её не штрафуем
    assert solution.relevant_quoted_names("что с «Меридиан»?", [trade, trading]) == set()


def test_topic_words_boost_body_and_section() -> None:
    common = "Общие условия."
    chunks = [
        {
            "doc": "ДОГОВОР АРЕНДЫ № А-08/2024",
            "number": "А-08/2024",
            "clause": "3.1",
            "section": "3. Арендная плата",
            "text": "Условие о плате.",
            "searchable": common,
        },
        {
            "doc": "ДОГОВОР АРЕНДЫ № А-08/2024",
            "number": "А-08/2024",
            "clause": "6.3",
            "section": "6. Изменение и расторжение договора",
            "text": "Возврат помещения.",
            "searchable": common,
        },
    ]
    retriever = solution.Retriever(chunks)

    result = retriever.search("какая арендная плата?", k=2)

    assert result
    assert result[0]["clause"] == "3.1"


def test_with_neighbours_adds_fragments_of_the_same_law_article() -> None:
    chunks = [
        {
            "doc": "Статья 164. Налоговые ставки",
            "clause": "ст. 164",
            "text": f"Часть статьи {index}.",
            "searchable": f"часть {index}",
        }
        for index in range(4)
    ]
    selected = [dict(chunks[1])]

    result = solution.with_neighbours(selected, chunks, limit=3)

    assert [chunk["text"] for chunk in result] == [
        "Часть статьи 1.",
        "Часть статьи 2.",
    ]


def test_with_neighbours_respects_budget() -> None:
    selected = [{"doc": "Д", "clause": "3.1", "text": "12345"}]
    chunks = selected + [{"doc": "Д", "clause": "3.2", "text": "67890"}]

    assert solution.with_neighbours(selected, chunks, max_extra_chars=4) == selected


def test_own_company_is_detected_from_corpus_not_hardcoded() -> None:
    chunks = [
        {
            "doc": f"ДОГОВОР № {index} с ООО «Вектор»",
            "text": "Поставка товара.",
            "searchable": f"ДОГОВОР № {index} с ООО «Вектор» Поставка товара.",
        }
        for index in range(3)
    ]
    chunks += [
        {
            "doc": "ДОГОВОР № 10 с ООО «Техпром»",
            "text": "Поставка товара.",
            "searchable": "ДОГОВОР № 10 с ООО «Техпром» Поставка товара.",
        },
        {
            "doc": "ДОГОВОР № 11 с ООО «Ромашка»",
            "text": "Поставка товара.",
            "searchable": "ДОГОВОР № 11 с ООО «Ромашка» Поставка товара.",
        },
    ]

    common = solution.common_quoted_names(chunks)
    assert "вектор" in common
    assert "вектор" in solution.Retriever(chunks).common_quoted_names
    assert "техпром" not in common
    assert solution.relevant_quoted_names("что с «Вектор»?", chunks) == set()
    assert solution.relevant_quoted_names("что с «Техпром»?", chunks) == {"техпром"}


def test_query_synonyms_find_initiator() -> None:
    expanded = solution.expand_query("кто может запустить согласование договора?")
    assert "инициатор" in expanded
    assert "согласован" in expanded
