"""Выбор предложения для ответа и для сниппета источника.

Отдельный модуль, потому что логика общая для двух мест: сниппет в
`sources` (бэкенд) и цитата в ответе (сабмит). Живёт здесь как
единственный источник правды, в сабмит попадает генерацией
(scripts/sync_submit.py).
"""
from __future__ import annotations

import re

# вопрос про конкретное значение: сумму, срок, процент
VALUE_MARKERS = (
    "сколько", "какая", "какой", "какое", "какие", "размер", "сумма", "сумму",
    "стоимость", "цена", "процент", "ставка", "тариф", "штраф", "пеня",
    "неустойка", "плата", "срок", "когда",
)
# вопрос про время: значение важно, но порядок не переставляем
TIMING_MARKERS = ("когда", "в какой срок", "как часто", "с какого", "до какого", "какого числа")


def has_amount(text: str) -> bool:
    """В тексте есть величина: сумма, процент или срок в днях.

    «не позднее 5-го числа» — это дата, а не значение: цифра есть, смысла нет.
    """
    if re.search(r"\d[\d\s]{2,}", text):
        return True
    if re.search(r"\d+[.,]\d+\s*%", text):
        return True
    if re.search(r"\d+\s*(?:%|процент|руб|дн)", text):
        return True
    return False


def wants_value(question: str) -> bool:
    """Спрашивают ли про конкретное значение (сумму, срок, процент)."""
    low = question.lower()
    return any(marker in low for marker in VALUE_MARKERS)


def asks_timing(question: str) -> bool:
    low = question.lower()
    return any(marker in low for marker in TIMING_MARKERS)


def pick_snippet(chunk: dict, question: str, limit: int = 300) -> str:
    """Сниппет источника: на вопрос о сумме первым идёт предложение с суммой.

    Раньше брался просто text[:300], и на «какая арендная плата» в источнике
    оказывался пункт про дату внесения без самой суммы.
    """
    text = str(chunk.get("text") or "").strip()
    if len(text) <= limit:
        return text

    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]
    if not sentences:
        return text[:limit]

    if wants_value(question) and not asks_timing(question):
        ranked = sorted(sentences, key=lambda s: (has_amount(s), len(s)), reverse=True)
        snippet = ranked[0]
    else:
        snippet = sentences[0]
    return snippet[:limit] if len(snippet) > limit else snippet
