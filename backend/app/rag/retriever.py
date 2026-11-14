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

    tokenized_query = tokenize(query_clean)
    scores = _bm25_index.get_scores(tokenized_query)

    sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    
    for idx, score in sorted_indices[:k]:
        if score == 0: 
            continue
        chunk = _chunks_store[idx].copy()
        chunk['score'] = float(score)
        results.append(chunk)
        
    return results

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

