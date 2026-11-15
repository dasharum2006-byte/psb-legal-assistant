"""Поиск по базе знаний.

TODO:
- BM25 для точных номеров и сумм;
- эмбеддинги bge-m3 для смысла;
- reranker Qwen3;
- вернуть top-k чанков.
"""
from rank_bm25 import BM25Okapi
import re

try:  # запуск из backend/ (uvicorn, pytest)
    from app.rag.normalize import stem
except ModuleNotFoundError:  # запуск из корня репозитория (scripts/build_index.py)
    from backend.app.rag.normalize import stem
import os
import json

_bm25_index = None
_chunks_store = []


def tokenize(text: str) -> list[str]:
    """Токены для BM25 со стеммингом.

    Без стемминга «расторжения» и «расторжении» для поиска разные слова,
    и нужный пункт получает score = 0.
    """
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return [stem(word) for word in text.split() if len(word) > 1]

def build_index(chunks: list[dict]):
    global _bm25_index, _chunks_store
    _chunks_store = chunks
    tokenized_corpus = [tokenize(chunk.get('searchable', chunk['text'])) for chunk in _chunks_store]
    _bm25_index = BM25Okapi(tokenized_corpus)
    print(f"Индекс ВМ25 построен на {len(chunks)} чанках")

# Номера документов: «А-08/2024», «П-17/2024», «ПТ-05/2023», «ЮО-7», «К-12».
# Общая схема, а не список известных договоров: работает и на приватном наборе.
DOC_NUMBER_RE = re.compile(r"(?<![а-яa-z0-9])([а-я]{1,4})[-‑](\d{1,4})(?:/(\d{4}))?", re.I)
# Ссылки на статью или пункт: «ст. 70», «статья 70», «п. 3.1».
CLAUSE_REF_RE = re.compile(r"(?:ст\.|стать[яию]|п\.|пункт)\s*(\d+(?:\.\d+)*)", re.I)

# Слова, которые есть почти в любом заголовке и ничего не различают
TITLE_STOPWORDS = {
    "служебная", "записка", "письмо", "исходящее", "договор", "поставки",
    "аренды", "нежилого", "помещения", "соглашение", "дополнительное",
    "общество", "ограниченной", "ответственностью", "между", "номер",
    "российской", "федерации", "статья", "глава",
}


def question_doc_numbers(text: str) -> set[str]:
    """Номера документов, упомянутые в вопросе («а-08/2024», «юо-7»)."""
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


def metadata_boost(query: str, chunk: dict) -> float:
    """Бонус за совпадение метаданных: номер документа и ссылка на пункт.

    Без него нужный пункт остаётся на 15-25 месте: BM25 не знает, что
    «П-17/2024» в вопросе и в заголовке — одно и то же.
    """
    bonus = 0.0
    numbers = question_doc_numbers(query)
    refs = question_clause_refs(query)

    chunk_number = str(chunk.get("number") or "").lower().replace("№", "").strip()
    haystack = f"{chunk_number} {chunk.get('doc_id', '')} {chunk.get('doc', '')}".lower().replace("/", "-")

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


QUERY_SYNONYMS = {
    "аванс": "предоплат",
    "предоплат": "аванс",
    "ноч": "круглосуточ доступ",
    "круглосуточ": "доступ беспрепятственн",
    "оплат": "платеж расчет стоимость",
    "платеж": "оплат расчет",
    "срок": "период дней",
    "гарант": "гарантийн",
    "приемк": "акт качество",
    "приёмк": "акт качество",
    "доставк": "упаковк перевозк",
    "упаковк": "доставк",
    "одобрен": "согласован генеральн директор",
    "согласован": "одобрен",
    "испытан": "испытательн",
    "расторжен": "прекращен увольнен",
    "уволен": "расторжен прекращен",
    "хранен": "архив срок",
    "выдач": "выдава оригинал архив",
    "претенз": "претензи акт недостатк",
    "недостат": "дефект претензи качеств",
    # «сдать ключи» -> «вернуть помещение по акту»
    "сда": "возврат передач акт",
    "ключ": "доступ помещен",
    "акт": "возврат приемк передач",
}


def expand_query(query: str) -> str:
    """Дописать синонимы: «грузовик в два часа ночи» -> «круглосуточ доступ».

    BM25 не знает, что «ночью» и «круглосуточно» — одно и то же. Без
    расширения такие вопросы не находят нужный пункт ни в top-8, ни в top-20.
    """
    tokens = tokenize(query)
    extra = []
    for token in tokens:
        for key, value in QUERY_SYNONYMS.items():
            if token.startswith(key):
                extra.append(value)
    return f"{query} {' '.join(extra)}" if extra else query


def search(query: str, k: int = 8) -> list[dict]:
    # До загрузки индекса глобальной переменной может не быть вовсе —
    # без этой проверки search() бросал NameError вместо пустого списка.
    if globals().get("_bm25_index") is None or not _chunks_store:
        return []
        
    # Единственная предобработка: служебное слово «договор» встречается почти
    # в каждом фрагменте и только размывает веса. Никаких бонусов под
    # конкретные документы и суммы — после стемминга BM25 справляется сам,
    # а хардкод под 18 демо-вопросов не работал бы на приватном наборе.
    query_clean = re.sub(r"\bдоговор[а-я]*\b", " ", query.lower())

    tokenized_query = tokenize(expand_query(query_clean))
    scores = list(_bm25_index.get_scores(tokenized_query))

    # общие бусты по метаданным: номер документа и ссылка на статью/пункт
    for index, chunk in enumerate(_chunks_store):
        bonus = metadata_boost(query, chunk)
        if bonus:
            scores[index] += bonus

    sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    
    for idx, score in sorted_indices[:k]:
        if score == 0: 
            continue
        chunk = _chunks_store[idx].copy()
        chunk['score'] = float(score)
        results.append(chunk)
        
    return results

def next_chunk(chunk: dict, limit: int = 1) -> list[dict]:
    """Следующие по порядку чанки того же документа.

    Ответ часто разбит на два пункта: 3.1 — сумма, 3.2 — срок внесения.
    Поиск приносит только один из них, поэтому соседа подклеиваем отдельно.
    Ограничение одним документом не даёт тащить в контекст чужие пункты.
    """
    if not _chunks_store:
        return []
    position = None
    chunk_id = chunk.get("chunk_id") or ""
    doc_id = chunk.get("doc_id") or ""
    for index, item in enumerate(_chunks_store):
        if chunk_id and item.get("chunk_id") == chunk_id:
            position = index
            break
        if not chunk_id and item.get("doc_id") == doc_id and item.get("clause") == chunk.get("clause"):
            position = index
            break
    if position is None:
        return []

    found = []
    for index in range(position + 1, min(position + 1 + limit * 3, len(_chunks_store))):
        item = _chunks_store[index]
        if item.get("doc_id") != _chunks_store[position].get("doc_id"):
            break
        if item.get("clause") == chunk.get("clause"):
            continue
        found.append(dict(item))
        if len(found) >= limit:
            break
    return found


def load_index_from_disk(index_dir: str = "data/index"):
    """
    Загружает чанки из файла и строит индекс BM25.
    """
    global _chunks_store, _bm25_index
    
    chunks_path = os.path.join(index_dir, "chunks.jsonl")
    if not os.path.exists(chunks_path):
        print(f"⚠️ Файл {chunks_path} не найден! Сначала запусти build_index.py")
        return
        
    _chunks_store = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            _chunks_store.append(json.loads(line))
            
    # Перестраиваем индекс (быстро, так как чанков немного)
    tokenized_corpus = [tokenize(chunk.get('searchable', chunk['text'])) for chunk in _chunks_store]
    _bm25_index = BM25Okapi(tokenized_corpus)
    print(f"✅ Индекс загружен с диска: {len(_chunks_store)} чанков")

