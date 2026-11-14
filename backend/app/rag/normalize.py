"""Нормализация текста для поиска и проверки ссылок"""

from __future__ import annotations

import re

_SEPARATORS = str.maketrans(dict.fromkeys("«»\"'`№", " "))
_AMOUNT = re.compile(
    # тысячи: «450 000» или «1.500.000»; дробь — только 1-2 знака
    # (десятые доли миллионов и копейки), иначе «1.500.000» распадалось
    # на число 1 и дробь .500 и сумма терялась
    r"(?<![\w.,/])(?P<number>\d{1,3}(?:[ .]\d{3})+|\d+)"
    r"(?P<fraction>[.,]\d{1,2})?(?!\d)"
    r"(?:\s*(?P<unit>"
    r"млрд|миллиард(?:а|ов|ами)?|"          # миллиарды
    r"млн|миллион(?:а|ов|ами)?|"            # миллионы
    r"тыс(?:яч(?:а|и|ами)?)?|т\.\s?(?:р|руб)\.?|"   # тысячи и «т.р.»
    r"руб(?:ль|ля|лей|лями)?|р\.|₽"         # рубли
    r")(?!\w))?"
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
        if unit.startswith(("млрд", "миллиард")):
            multiplier = 1_000_000_000
        elif unit.startswith(("млн", "миллион")):
            multiplier = 1_000_000
        elif unit.startswith(("тыс", "т.р", "т. р")):
            multiplier = 1_000

        fraction = (match["fraction"] or "")[1:]
        if len(fraction) > 2:
            fraction = ""
        if fraction and multiplier == 1 and not unit:
            continue
        value = int(re.sub(r"[ .]", "", number)) * multiplier
        if fraction:
            value += int(fraction) * multiplier // 10 ** len(fraction)
        amounts.append(value)
    return amounts


_ARTICLE_RE = re.compile(r"(?:ст\.|стать[яиею])\s*(\d+(?:\.\d+)*)")
_PART_RE = re.compile(r"(?:ч\.|част(?:ь|и))\s*(\d+)")
_POINT_RE = re.compile(r"(?:п\.|пункт(?:а|е|ом)?)\s*(\d+(?:\.\d+)*)")


def clause_key(text: str) -> str | None:
    """Канонический ключ ссылки, без коллизий.

    «ст. 15 ч. 2» -> «15#2»   (часть статьи)
    «ст. 15.1»    -> «15.1»   (отдельная статья Кодекса)
    «п. 3.1»      -> «3.1»

    normalize_clause() склеивала часть через точку и давала одинаковый
    результат для «ст. 15.1» и «ст. 15 ч. 2».
    """
    if not text:
        return None
    low = normalize_text(text)

    article = _ARTICLE_RE.search(low)
    part = _PART_RE.search(low)
    if article:
        return f"{article.group(1)}#{part.group(1)}" if part else article.group(1)

    point = _POINT_RE.search(low)
    if point:
        return point.group(1)
    if part:
        return f"#{part.group(1)}"

    # чанкер хранит номер без префикса: clause = "3.1" или "ст. 70"
    bare = re.fullmatch(r"\s*(\d+(?:\.\d+)*)\s*", low)
    if bare:
        return bare.group(1)
    return None


def clause_keys(text: str) -> set[str]:
    """Все ключи ссылок, встречающихся в тексте (для проверки ответа)."""
    if not text:
        return set()
    low = normalize_text(text)
    keys = {match.group(1) for match in _POINT_RE.finditer(low)}
    keys |= {match.group(1) for match in _ARTICLE_RE.finditer(low)}
    return keys


def keys_match(answer_key: str, chunk_key: str) -> bool:
    """Ссылка из ответа подтверждается чанком (или его более общим уровнем)."""
    if not answer_key or not chunk_key:
        return False
    if answer_key == chunk_key:
        return True
    # «ст. 15» подтверждается чанком «ст. 15.1» (ссылка на статью целиком)
    return chunk_key.startswith(answer_key + ".") or answer_key.startswith(chunk_key + "#")


def normalize_clause(text: str) -> str | None:
    """Извлечь номер первого пункта, статьи или части."""
    match = _CLAUSE.search(normalize_text(text))
    if match is None:
        return None
    article, article_part, clause, part = match.groups()
    if article and article_part:
        return f"{article}.{article_part}"
    return article or clause or part
