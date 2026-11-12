"""Нормализация текста для поиска и проверки ссылок"""

from __future__ import annotations

import re

_SEPARATORS = str.maketrans(dict.fromkeys("«»\"'`№", " "))
_AMOUNT = re.compile(
    r"(?<![\w.,/])(?P<number>\d{1,3}(?: \d{3})+|\d+)"
    r"(?P<fraction>[.,]\d+)?"
    r"(?:\s*(?P<unit>млн|миллион(?:а|ов)?|тыс(?:яч(?:а|и)?)?|"
    r"руб(?:ль|ля|лей)?|р\.|₽)(?!\w))?"
    r"(?P<percent>\.?\s*(?:%|процент(?:а|ов)?\b))?"
)
_NUMBER = r"\d+(?:\.\d+)*"
_PART = rf"(?:ч\.|част(?:ь|и))\s*({_NUMBER})"
_CLAUSE = re.compile(
    rf"(?<!\w)(?:(?:ст\.|стать[яиею])\s*({_NUMBER})"
    rf"(?:\s*,?\s*{_PART})?|(?:п\.|пункт(?:а|е|ом)?)\s*({_NUMBER})|{_PART})"
)


def normalize_text(text: str) -> str:
    """Привести текст к сравнимому виду."""
    return " ".join(text.lower().replace("ё", "е").translate(_SEPARATORS).split())


def tokens(text: str) -> list[str]:
    """Разбить текст на токены, сохраняя дефисы внутри обозначений."""
    return re.findall(r"[^\W_]+(?:-+[^\W_]+)*", normalize_text(text))


def parse_amounts(text: str) -> list[int]:
    """Извлечь целые суммы в рублях, пропуская проценты и годы."""
    amounts = []
    for match in _AMOUNT.finditer(normalize_text(text)):
        number = match["number"]
        unit = match["unit"] or ""
        if match["percent"] or (len(number) == 4 and number.isdigit() and not unit):
            continue

        multiplier = 1
        if unit == "млн" or unit.startswith("миллион"):
            multiplier = 1_000_000
        elif unit.startswith("тыс"):
            multiplier = 1_000

        fraction = (match["fraction"] or "")[1:]
        if fraction and multiplier == 1 and not unit:
            continue
        value = int(number.replace(" ", "")) * multiplier
        if fraction:
            value += int(fraction) * multiplier // 10 ** len(fraction)
        amounts.append(value)
    return amounts


def normalize_clause(text: str) -> str | None:
    """Извлечь номер первого пункта, статьи или части."""
    match = _CLAUSE.search(normalize_text(text))
    if match is None:
        return None
    article, article_part, clause, part = match.groups()
    if article and article_part:
        return f"{article}.{article_part}"
    return article or clause or part
