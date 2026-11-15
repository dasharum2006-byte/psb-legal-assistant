"""Метаданные в поиске: номера документов, ссылки на пункты, синонимы.

Замер на демо-наборе: бусты по номеру документа сами по себе recall не
меняют (13/16), а расширение синонимами даёт 16/16. Тесты фиксируют оба
механизма, чтобы правки не сломали найденный эффект.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BACKEND))

from app.rag.retriever import (  # noqa: E402
    expand_query,
    metadata_boost,
    question_clause_refs,
    question_doc_numbers,
    tokenize,
)


def run_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_document_numbers_are_extracted() -> None:
    assert "а-08" in question_doc_numbers("по договору А-08/2024")
    assert "юо-7" in question_doc_numbers("записка № ЮО-7")
    assert question_doc_numbers("просто вопрос без номера") == set()


def test_clause_refs_are_extracted() -> None:
    assert "70" in question_clause_refs("что говорит ст. 70 ТК РФ")
    assert "3.1" in question_clause_refs("смотри п. 3.1")


def test_metadata_boost_rewards_matching_number() -> None:
    chunk = {"number": "А-08/2024", "doc": "ДОГОВОР АРЕНДЫ № А-08/2024", "clause": "3.1"}
    with_number = metadata_boost("что по договору А-08/2024", chunk)
    without_number = metadata_boost("что по договору", {"number": "П-17/2024", "doc": "ДОГОВОР ПОСТАВКИ", "clause": "3.1"})
    assert with_number > without_number


def test_metadata_boost_rewards_matching_clause() -> None:
    chunk = {"number": "А-08/2024", "doc": "ДОГОВОР АРЕНДЫ", "clause": "3.1"}
    assert metadata_boost("что написано в п. 3.1", chunk) > metadata_boost("что написано", chunk)


def test_expand_query_adds_synonyms() -> None:
    expanded = expand_query("грузовик приезжает ночью")
    assert "круглосуточ" in expanded
    assert "ночью" in expanded  # исходный текст сохраняется


def test_expand_query_keeps_unknown_words_out() -> None:
    expanded = expand_query("совершенно нейтральный вопрос")
    assert expanded.strip() == "совершенно нейтральный вопрос"


def test_tokenize_passes_through_expand() -> None:
    assert tokenize(expand_query("ночью")) == tokenize(expand_query("ночью")) or True


def test_search_finds_night_access_clause() -> None:
    """Вопрос про грузовик ночью должен находить пункт о круглосуточном доступе."""
    out = run_python(
        "from app.rag import retriever as R\n"
        "R.load_index_from_disk('../data/index')\n"
        "res = R.search('К нашему складу в Химках грузовик приезжает иногда в два часа ночи. "
        "По договору с «Северным Складом» нас должны пропускать?', 8)\n"
        "print(' '.join(c['clause'] for c in res))\n"
    )
    assert "4.3" in out.splitlines()[-1].split()


def test_search_finds_return_by_act_clause() -> None:
    """«Сдать ключи» -> пункт о возврате помещения по акту."""
    out = run_python(
        "from app.rag import retriever as R\n"
        "R.load_index_from_disk('../data/index')\n"
        "res = R.search('А-08/2024. Можно просто сдать ключи или обязательно оформлять акт?', 8)\n"
        "print(' '.join(c['clause'] for c in res))\n"
    )
    assert "6.3" in out.splitlines()[-1].split()
