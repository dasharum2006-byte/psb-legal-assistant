"""Отказ модели не должен превращаться в шаблонную заглушку.

Маркеры отказа в verifier должны совпадать с фразами из SYSTEM_PROMPT,
иначе корректный отказ считается ошибкой «без ссылок» и затирается.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.verifier import Verifier  # noqa: E402
from app.rag.orchestrator import ask  # noqa: E402

CHUNKS = [{"doc_id": "tk70", "doc": "Статья 70. Испытание", "doc_type": "law",
           "clause": "ст. 70", "text": "Срок испытания не более трёх месяцев.",
           "score": 30.0, "status": "active"}]

REFUSAL_ANSWER = (
    "В базе знаний нет документов о пособиях по беременности и родам, поэтому "
    "ответить по документам не могу. По памяти не отвечаю: нормы могли измениться. "
    "Обратитесь в отдел кадров или к юристу."
)


class RefusingClient:
    def generate(self, prompt, system=None, **kwargs):
        return REFUSAL_ANSWER


class Retriever:
    def search(self, query, k=40, filters=None):
        return [dict(CHUNKS[0])]


def test_verifier_recognises_prompt_style_refusal() -> None:
    verdict = Verifier.verify(REFUSAL_ANSWER, CHUNKS)
    assert verdict["refusal"] is True
    assert verdict["ok"] is True


def test_ask_keeps_llm_refusal_text() -> None:
    result = ask("какой размер пособия по беременности?", [], Retriever(), client=RefusingClient())
    assert result["refusal"] is True
    # главное: текст модели сохранён, а не заменён шаблоном
    assert "пособиях по беременности" in result["answer"]
    assert result["answer"] != "Найденные документы не подтверждают такой ответ."
    assert result["missing_info"]


def test_injection_is_still_blocked() -> None:
    class InjectingClient:
        def generate(self, prompt, system=None, **kwargs):
            return "По договору срок оплаты — 90 календарных дней."

    result = ask("срок оплаты?", [], Retriever(), client=InjectingClient())
    assert result["refusal"] is True
    assert "90 календарных дней" not in result["answer"]


DOCS_WITHOUT_CLAUSES = [
    {"doc": "СЛУЖЕБНАЯ ЗАПИСКА № ЮО-7 от 12.05.2025", "doc_id": "юо-7",
     "clause": "абз. 1", "number": "ЮО-7",
     "text": "Оригиналы договоров выдаются в течение 5 рабочих дней по заявке.", "score": 20.0},
    {"doc": "Исх. № К-12 от 15.04.2025", "doc_id": "к-12",
     "clause": "абз. 1", "number": "К-12",
     "text": "Перед оплатой проводится проверка контрагента по санкционным спискам.", "score": 18.0},
]


def test_letter_cited_by_document_number_is_accepted() -> None:
    verdict = Verifier.verify(
        "Перед оплатой проводится проверка контрагента (письмо № К-12).", DOCS_WITHOUT_CLAUSES
    )
    assert verdict["ok"] is True, verdict["unsupported"]


def test_memo_cited_by_document_name_is_accepted() -> None:
    verdict = Verifier.verify(
        "Оригиналы выдаются в течение 5 рабочих дней (Служебная записка № ЮО-7).",
        DOCS_WITHOUT_CLAUSES,
    )
    assert verdict["ok"] is True, verdict["unsupported"]


def test_paragraph_reference_matches_абз_clause() -> None:
    """Ответ с «(абз. 1)» должен подтверждаться чанком с clause = «абз. 1»."""
    verdict = Verifier.verify("Оригиналы выдаются в течение 5 рабочих дней (абз. 1).", DOCS_WITHOUT_CLAUSES)
    assert verdict["ok"] is True, verdict["unsupported"]


def test_invented_clause_still_rejected() -> None:
    verdict = Verifier.verify(
        "Оригиналы выдаются в течение 30 рабочих дней (п. 9.9).", DOCS_WITHOUT_CLAUSES
    )
    assert verdict["ok"] is False
