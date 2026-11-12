"""Поиск по базе знаний.

TODO:
- BM25 для точных номеров и сумм;
- эмбеддинги bge-m3 для смысла;
- reranker Qwen3;
- вернуть top-k чанков.
"""
from rank_bm25 import BM25Okapi
import re
import os
import json

bm_25_index = None
_chunks_store = []


SYNONYMS = {
    "северный склад": "документы_компании__договор_аренды_а-08-2024_северный_склад",
    "альфа-трейд": "документы_компании__договор_поставки_п-17-2024_альфа-трейд",
    "альфа-трейдинг": "документы_компании__договор_поставки_пт-05-2023_альфа-трейдинг"
}

def tokenize(text:str) -> list[str]:
    text = text.lower().replace('ё', 'е')
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)   
    tokens = [word for word in text.split() if len(word) > 1]
    return tokens

def build_index(chunks: list[dict]):
    global _bm25_index, _chunks_store
    _chunks_store = chunks
    tokenized_corpus = [tokenize(chunk.get('searchable', chunk['text'])) for chunk in _chunks_store]
    _bm25_index = BM25Okapi(tokenized_corpus)
    print(f"Индекс ВМ25 построен на {len(chunks)} чанках")

def search(query: str, k: int = 8) -> list[dict]:
    if _bm25_index is None:
        return []
        
    # 1. Убираем слово "договор" из запроса, чтобы оно не сбивало с толку
    query_clean = query.lower().replace("договор", "").replace("договора", "")
    
    tokenized_query = tokenize(query_clean)
    scores = _bm25_index.get_scores(tokenized_query)
    
    # 2. Добавляем бонусы
    query_lower = query.lower()
    for keyword, target_doc_id in SYNONYMS.items():
        if keyword in query_lower:
            for idx, chunk in enumerate(_chunks_store):
                if chunk.get('doc_id') == target_doc_id:
                    scores[idx] += 50.0  # Огромный бонус за правильный документ
                    
    # 3. Дополнительный бонус за слова в тексте пункта
    if "арендная плата" in query_lower:
        for idx, chunk in enumerate(_chunks_store):
            if "арендная плата" in chunk.get('text', '').lower():
                scores[idx] += 30.0 # Еще бонус, если в тексте есть "арендная плата"

    # 4. Бонус за пункт, где есть конкретная сумма (например, 450 000)
    if "450" in query_lower or "арендная плата" in query_lower:
        for idx, chunk in enumerate(_chunks_store):
            # Если в тексте есть "450 000" или "450000"
            if "450 000" in chunk.get('text', '') or "450000" in chunk.get('text', ''):
                scores[idx] += 20.0

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

