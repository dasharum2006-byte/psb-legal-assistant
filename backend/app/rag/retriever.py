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
    from app.rag.metadata import (
        CLAUSE_REF_RE,
        DOC_NUMBER_RE,
        TITLE_STOPWORDS,
        common_quoted_names,
        exact_mention_boost,
        foreign_name_penalty,
        has_past_year,
        metadata_boost,
        question_clause_refs,
        question_doc_numbers,
        question_names,
        relevant_quoted_names,
        repealed_penalty,
        title_keywords,
    )
except ModuleNotFoundError:  # запуск из корня репозитория (scripts/build_index.py)
    from backend.app.rag.normalize import stem
    from backend.app.rag.metadata import (
        CLAUSE_REF_RE,
        DOC_NUMBER_RE,
        TITLE_STOPWORDS,
        common_quoted_names,
        exact_mention_boost,
        foreign_name_penalty,
        has_past_year,
        metadata_boost,
        question_clause_refs,
        question_doc_numbers,
        question_names,
        relevant_quoted_names,
        repealed_penalty,
        title_keywords,
    )
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


# Служебные слова шапки: есть почти в каждом пункте и не должны считаться
# смыслом вопроса.
META_NOISE_TOKENS = set(tokenize("договор поставка аренда регламент записка письмо ооо"))
_body_tokens: list[set[str]] = []
_section_tokens: list[set[str]] = []
_common_names: set[str] = set()


def _refresh_token_cache() -> None:
    global _body_tokens, _section_tokens, _common_names
    _body_tokens = [set(tokenize(str(c.get("text") or ""))) for c in _chunks_store]
    _section_tokens = [set(tokenize(str(c.get("section") or ""))) for c in _chunks_store]
    _common_names = common_quoted_names(_chunks_store)


def _last_user_message(text: str) -> str:
    """Последняя реплика пользователя, если пришла история диалога."""
    parts = re.split(r"(?i)(пользователь|ассистент)\s*:", text)
    last = text
    for index in range(1, len(parts) - 1, 2):
        if parts[index].lower() == "пользователь" and parts[index + 1].strip():
            last = parts[index + 1].strip()
    return last


def all_chunks() -> list[dict]:
    """Все чанки индекса для независимого семантического поиска.

    BM25 отдаёт только top-k, поэтому перефразированный пункт, которого нет
    в этом top-k, иначе не увидит ни эмбеддинг, ни реранкер. Возвращаем копию
    списка (dict'ы не копируем глубоко: semantic их не меняет).
    """
    return list(_chunks_store)


def build_index(chunks: list[dict]):
    global _bm25_index, _chunks_store
    _chunks_store = chunks
    tokenized_corpus = [tokenize(chunk.get('searchable', chunk['text'])) for chunk in _chunks_store]
    # пустой корпус — валидное состояние (архив не прочитался), BM25 на нём падает
    _bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    _refresh_token_cache()
    print(f"Индекс ВМ25 построен на {len(chunks)} чанках")


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
    "инициатор": "запуст согласован",
    "запуст": "инициатор согласован",
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

    # общие бусты по метаданным: номер документа, точное название,
    # ссылка на статью/пункт; штрафы за чужую компанию и старую редакцию
    relevant_names = relevant_quoted_names(query, _chunks_store, _common_names)
    if len(_body_tokens) != len(_chunks_store):
        _refresh_token_cache()

    # Смысловые слова — из последней реплики пользователя, без шапки и
    # названий документов, иначе «договор»/«аренда» поднимают случайные пункты.
    topic_text = expand_query(_last_user_message(query))
    quoted_tokens: set[str] = set()
    for name in question_names(query):
        quoted_tokens |= set(tokenize(name))
    topic_tokens = set(tokenize(topic_text)) - quoted_tokens - META_NOISE_TOKENS

    for index, chunk in enumerate(_chunks_store):
        bonus = (
            metadata_boost(query, chunk)
            + repealed_penalty(query, chunk)
            + foreign_name_penalty(query, chunk, relevant_names)
        )
        if topic_tokens:
            body_hits = len(_body_tokens[index] & topic_tokens)
            section_hits = len(_section_tokens[index] & topic_tokens)
            # одно общее слово вроде «размер» ещё не делает пункт ответом
            if body_hits + section_hits >= 2:
                bonus += 5.0 * body_hits + 7.0 * section_hits
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
    # пустой корпус — валидное состояние (архив не прочитался), BM25 на нём падает
    _bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    _refresh_token_cache()
    print(f"✅ Индекс загружен с диска: {len(_chunks_store)} чанков")

