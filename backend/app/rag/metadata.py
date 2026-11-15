"""Метаданные поиска: номера документов и ссылки на пункты.

Общий модуль для бэкенда и зачётного solution.py. Вынесен отдельно, чтобы
одна и та же логика бустов не расходилась между двумя конвейерами:
`scripts/sync_submit.py` вставляет его в solution.py как GENERATED:metadata.
"""
from __future__ import annotations

import re
from collections import Counter

# Номера документов: «А-08/2024», «П-17/2024», «ПТ-05/2023», «ЮО-7», «К-12».
# Общая схема, а не список известных договоров: работает и на приватном наборе.
DOC_NUMBER_RE = re.compile(r"(?<![а-яa-z0-9])([а-я]{1,4})[-‑](\d{1,4})(?:/(\d{4}))?", re.I)
# Ссылки на статью или пункт: «ст. 70», «статья 70», «п. 3.1».
CLAUSE_REF_RE = re.compile(r"(?:ст\.|стать[яию]|п\.|пункт)\s*(\d+(?:\.\d+)*)", re.I)
# Названия в кавычках: «Альфа-Трейд», "Альфа-Трейдинг".
QUOTED_NAME_RE = re.compile(r"[«\"“]([^«»\"“”]{2,80})[»\"”]")
# Прошлые годы: вопрос про редакцию на дату — старые редакции не штрафуем.
PAST_YEAR_RE = re.compile(r"(?<!\d)20(?:20|21|22|23|24)(?!\d)")

# Слова, которые есть почти в любом заголовке и ничего не различают
TITLE_STOPWORDS = {
    "служебная", "записка", "письмо", "исходящее", "договор", "поставки",
    "аренды", "нежилого", "помещения", "соглашение", "дополнительное",
    "общество", "ограниченной", "ответственностью", "между", "номер",
    "российской", "федерации", "статья", "глава",
}

# Своя компания определяется по базе, а не по названию из демо.
COMMON_NAME_SHARE = 0.40
COMMON_NAME_MIN_DOCS = 2
FOREIGN_NAME_PENALTY = -3.0


def question_doc_numbers(text: str) -> set[str]:
    """Номера документов, упомянутые в вопросе («а-08/2024», «юо-7»).

    Работает и без знака «№»: «Что сказано в записке ЮО-7?».
    """
    found = set()
    for letter, digits, year in DOC_NUMBER_RE.findall(text):
        found.add(f"{letter}-{digits}".lower())
        if year:
            found.add(f"{letter}-{digits}/{year}".lower())
    return found


def question_clause_refs(text: str) -> set[str]:
    """Ссылки на статью/пункт из вопроса («70», «3.1»)."""
    return {match.group(1) for match in CLAUSE_REF_RE.finditer(text)}


def title_keywords(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[а-яa-z]{5,}", text.lower().replace("ё", "е"))
        if word not in TITLE_STOPWORDS
    }


def question_names(text: str) -> set[str]:
    """Названия в кавычках, которые надо различить точно.

    «Альфа-Трейд» и «Альфа-Трейдинг» — разные контрагенты, поэтому
    сравнивать их подстрокой нельзя.
    """
    names = set()
    for raw in QUOTED_NAME_RE.findall(text):
        name = re.sub(r"\s+", " ", raw.lower().replace("ё", "е")).strip()
        if len(name) >= 2:
            names.add(name)
    return names


def _contains_exact(text: str, phrase: str) -> bool:
    """Фраза совпала целиком, а не как начало другого слова."""
    if not phrase:
        return False
    return re.search(
        rf"(?<![а-яa-z0-9]){re.escape(phrase)}(?![а-яa-z0-9])", text, re.I
    ) is not None


def has_past_year(text: str) -> bool:
    """Есть ли в вопросе 2020–2024 как дата, а не год в номере договора."""
    without_numbers = DOC_NUMBER_RE.sub(" ", text)
    without_numbers = re.sub(r"\b\d{1,4}/\d{4}\b", " ", without_numbers)
    return PAST_YEAR_RE.search(without_numbers) is not None


def exact_mention_boost(query: str, chunk: dict) -> float:
    """+25 за точное название в кавычках или номер документа.

    Номер может быть без «№»: «П-17/2024», «ЮО-7». Границы не дают
    «Альфа-Трейд» совпасть с «Альфа-Трейдинг».
    """
    haystack = " ".join(
        str(chunk.get(key) or "")
        for key in ("number", "doc_id", "doc", "title", "text")
    ).lower().replace("ё", "е")
    flat_haystack = haystack.replace("/", "-")

    for name in question_names(query):
        if _contains_exact(haystack, name):
            return 25.0
    for number in question_doc_numbers(query):
        if _contains_exact(flat_haystack, number.replace("/", "-")):
            return 25.0
    return 0.0


def is_repealed(chunk: dict) -> bool:
    return bool(chunk.get("repealed")) or str(chunk.get("status") or "").lower() == "repealed"


def common_quoted_names(chunks: list[dict]) -> set[str]:
    """Названия, которые встречаются почти во всех документах базы.

    Это «своя компания» (заказчик), её нельзя считать чужим договором.
    Частоту считаем по уникальным документам, а не по чанкам: у длинного
    договора иначе был бы несправедливый перевес.
    """
    documents: dict[str, dict] = {}
    for chunk in chunks:
        key = str(
            chunk.get("doc")
            or chunk.get("title")
            or chunk.get("doc_id")
            or id(chunk)
        )
        documents.setdefault(key, chunk)

    total = len(documents)
    if not total:
        return set()

    counts: Counter[str] = Counter()
    for chunk in documents.values():
        head = f"{chunk.get('doc') or ''} {str(chunk.get('text') or '')[:300]}"
        counts.update(question_names(head))

    threshold = max(COMMON_NAME_MIN_DOCS, int(total * COMMON_NAME_SHARE))
    common = {
        name for name, count in counts.items()
        if count >= threshold
    }
    if counts:
        top, count = counts.most_common(1)[0]
        second = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        # доминирующее название тоже считаем своим, даже если документов мало
        if count >= 3 and count >= 2 * max(1, second):
            common.add(top)
    return common


def relevant_quoted_names(
    query: str,
    chunks: list[dict],
    common_names: set[str] | None = None,
) -> set[str]:
    """Названия из вопроса, которые реально встречаются в заголовках базы.

    Так «круглосуточный доступ» в кавычках не считается названием компании
    и не штрафует все чужие договоры. Своя компания вычисляется по базе
    один раз в Retriever.__init__ и передаётся сюда.
    """
    common = common_quoted_names(chunks) if common_names is None else common_names
    wanted = {name for name in question_names(query) if name not in common}
    if not wanted:
        return set()

    known: set[str] = set()
    for chunk in chunks:
        title = f"{chunk.get('doc') or ''} {chunk.get('title') or ''}"
        known |= wanted & question_names(title)
        if known == wanted:
            break
    return known


def foreign_name_penalty(
    query: str, chunk: dict, relevant_names: set[str] | None = None
) -> float:
    """Слегка штрафуем чужой договор, если спросили про конкретную компанию."""
    wanted = relevant_names
    if wanted is None:
        wanted = set(question_names(query))
    if not wanted:
        return 0.0

    title = f"{chunk.get('doc') or ''} {chunk.get('title') or ''}".lower().replace("ё", "е")
    if any(_contains_exact(title, name) for name in wanted):
        return 0.0
    title_names = set(question_names(title))
    if title_names and not (title_names & wanted):
        return FOREIGN_NAME_PENALTY
    return 0.0


def repealed_penalty(query: str, chunk: dict) -> float:
    """-12 устаревшей редакции, если вопрос не про прошлый год."""
    if not is_repealed(chunk):
        return 0.0
    return 0.0 if has_past_year(query) else -12.0


def metadata_boost(query: str, chunk: dict) -> float:
    """Бонус за совпадение метаданных: номер документа и ссылка на пункт.

    Без него нужный пункт остаётся на 15-25 месте: BM25 не знает, что
    «П-17/2024» в вопросе и в заголовке — одно и то же.
    """
    bonus = exact_mention_boost(query, chunk)
    numbers = question_doc_numbers(query)
    refs = question_clause_refs(query)

    chunk_number = str(chunk.get("number") or "").lower().replace("№", "").strip()
    haystack = f"{chunk_number} {chunk.get('doc_id', '')} {chunk.get('doc', '')}".lower().replace("/", "-")

    # +25 за точное название/номер; старый +15 остаётся запасным путём,
    # если номер в документе записан в другой форме.
    if not bonus:
        for number in numbers:
            flat = number.replace("/", "-")
            if flat and flat in haystack:
                bonus += 15.0
                break
    if refs:
        clause = str(chunk.get("clause") or "").lower()
        core = clause.replace("ст.", "").replace("п.", "").strip()
        if core and any(ref == core or core.startswith(ref + ".") for ref in refs):
            bonus += 10.0

    matched = title_keywords(query) & title_keywords(str(chunk.get("doc") or ""))
    bonus += 3.0 * len(matched)
    return bonus
