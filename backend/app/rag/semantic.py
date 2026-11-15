"""Семантический поиск и реранк для бэкенда.

Платформенный прокси отдаёт /embeddings (bge-m3) и /rerank (cohere/rerank-5).
Локально этих эндпоинтов может не быть — тогда поиск молча остаётся на BM25:
деградация не должна ломать демо.

Логика выровнена с зачётным сабмитом:
  * эмбеддинги считаются батчами и кэшируются по модели;
  * семантика не только переупорядочивает BM25, но и добавляет кандидатов,
    которых BM25 не нашёл (это главный прирост на перефразированных вопросах);
  * /rerank получает top-N, а не весь список, и при сбое оставляет порядок;
  * /embeddings и /rerank включаются/отключаются независимо, поэтому
    недоступный реранкер не убивает семантический поиск и наоборот.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time

import httpx

logger = logging.getLogger(__name__)

EMB_MODEL = os.getenv("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cohere/rerank-5-fast")
SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "1200"))
EMBED_BATCH = max(1, int(os.getenv("EMBED_BATCH", "64")))
TIMEOUT = float(os.getenv("SEMANTIC_TIMEOUT", "20"))
# эндпоинт может подняться позже (сеть, прокси) — через минуту пробуем снова
DISABLE_SECONDS = float(os.getenv("SEMANTIC_DISABLE_SECONDS", "60"))

# ключ кэша — (модель, текст): если EMB_MODEL поменяется, старые векторы
# не будут молча использоваться как новые.
_CACHE: dict[tuple[str, str], list[float]] = {}
_disabled_until: dict[str, float] = {}
# Потоки FastAPI не должны одновременно считать одни и те же 875 чанков.
_EMBED_LOCK = threading.Lock()


def base_url() -> str:
    return (
        os.getenv("LLM_BASE_URL")
        or os.getenv("LLM_API_BASE")
        or os.getenv("BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).rstrip("/")


def api_key(endpoint: str = "embeddings") -> str:
    """У эмбеддера и реранкера могут быть разные ключи."""
    if endpoint == "rerank":
        return (
            os.getenv("RERANKER_API_KEY")
            or os.getenv("EMBEDDER_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        )
    return os.getenv("EMBEDDER_API_KEY") or os.getenv("LLM_API_KEY") or ""


def _configured(endpoint: str) -> bool:
    return bool(base_url() and api_key(endpoint)) and time.monotonic() >= _disabled_until.get(endpoint, 0.0)


def embeddings_available() -> bool:
    return _configured("embeddings")


def rerank_available() -> bool:
    return _configured("rerank")


def available() -> bool:
    """Есть ли хотя бы один из платформенных поисковых эндпоинтов."""
    return embeddings_available() or rerank_available()


def _disable(endpoint: str, exc: BaseException) -> None:
    _disabled_until[endpoint] = time.monotonic() + DISABLE_SECONDS
    logger.warning(
        "SEMANTIC_%s_UNAVAILABLE: %s: %s (повторим через %.0f с)",
        endpoint.upper(), type(exc).__name__, exc, DISABLE_SECONDS,
    )


def _make_client() -> httpx.Client:
    """Клиент httpx, устойчивый к socks-прокси в окружении.

    httpx не умеет SOCKS без доп. пакета и падает на этапе создания клиента.
    Тогда берём http-прокси из HTTPS_PROXY/HTTP_PROXY или идём без прокси.
    """
    try:
        return httpx.Client(timeout=TIMEOUT)
    except Exception:
        pass

    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = (os.getenv(name) or "").strip()
        if value.startswith(("http://", "https://")):
            try:
                return httpx.Client(timeout=TIMEOUT, proxy=value)
            except Exception:
                continue

    return httpx.Client(timeout=TIMEOUT, trust_env=False)


def _post(path: str, payload: dict, key: str, client: httpx.Client | None = None) -> dict:
    owns_client = client is None
    client = client or _make_client()
    try:
        for attempt in (1, 2):
            try:
                response = client.post(
                    base_url() + path,
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
                # временные ошибки прокси/лимиты — одна быстрая повторная попытка
                if response.status_code in (429, 500, 502, 503, 504) and attempt == 1:
                    time.sleep(0.5)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                if attempt == 1:
                    continue
                raise
        raise RuntimeError("эндпоинт не ответил")
    finally:
        if owns_client:
            client.close()


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _chunk_text(chunk: dict) -> str:
    """Текст для эмбеддинга: заголовок документа + раздел + пункт.

    В индексе это поле `searchable`. Без заголовка и раздела эмбеддинг
    пункта «3.1» не понимает, о какой теме идёт речь.
    """
    text = str(chunk.get("searchable") or "").strip()
    if not text:
        text = " ".join(
            str(chunk.get(key) or "")
            for key in ("doc", "title", "section", "clause", "text")
        ).strip()
    return text[:SNIPPET_CHARS] or " "


def embed(texts: list[str]) -> list[list[float]] | None:
    """Потокобезопасный вход в кэш эмбеддингов."""
    with _EMBED_LOCK:
        return _embed_serial(texts)


def _embed_serial(texts: list[str]) -> list[list[float]] | None:
    """Векторы для текстов; None — если эндпоинт недоступен.

    Батчи по EMBED_BATCH и кэш на процесс: при первом запросе считаются
    векторы всех чанков, дальше /embeddings отвечает только за вопрос.
    """
    if not texts or not embeddings_available():
        return None

    prepared = [str(text)[:SNIPPET_CHARS] or " " for text in texts]
    unique = list(dict.fromkeys(prepared))
    missing = [text for text in unique if (EMB_MODEL, text) not in _CACHE]

    if missing:
        try:
            with _make_client() as client:
                for start in range(0, len(missing), EMBED_BATCH):
                    batch = missing[start:start + EMBED_BATCH]
                    data = _post("/embeddings", {
                        "model": EMB_MODEL,
                        "input": batch,
                        "encoding_format": "float",
                    }, api_key("embeddings"), client=client)
                    for position, item in enumerate(data["data"]):
                        index = int(item.get("index", position))
                        text = batch[index] if 0 <= index < len(batch) else batch[position]
                        _CACHE[(EMB_MODEL, text)] = _normalize(
                            [float(x) for x in item["embedding"]]
                        )
        except Exception as exc:
            _disable("embeddings", exc)
            return None

    vectors = [_CACHE.get((EMB_MODEL, text)) for text in prepared]
    if any(vector is None for vector in vectors):
        return None
    return vectors  # type: ignore[return-value]


def warm_embeddings(chunks: list[dict]) -> int:
    """Прогреть кэш всех чанков до первых запросов пользователей."""
    if not chunks or not embeddings_available():
        return 0
    vectors = embed([_chunk_text(chunk) for chunk in chunks])
    return len(vectors) if vectors else 0


def semantic_candidates(
    question: str,
    chunks: list[dict],
    k: int = 12,
    min_score: float = 0.45,
) -> tuple[list[dict], float]:
    """Независимый поиск по смыслу среди всех чанков.

    Возвращает (кандидаты, лучшая близость). Кандидаты нужны не только для
    переупорядочивания: BM25 часто не находит перефразированный пункт, а
    эмбеддинг находит. Второй элемент позволяет отличить «ничего не подходит»
    от «подходит» и честно отказать.
    """
    if not chunks or not embeddings_available():
        return [], 0.0

    vectors = embed([question] + [_chunk_text(chunk) for chunk in chunks])
    if not vectors or len(vectors) != len(chunks) + 1:
        return [], 0.0

    query = vectors[0]
    scored = [
        (sum(a * b for a, b in zip(query, vector)), chunk)
        for vector, chunk in zip(vectors[1:], chunks)
    ]
    scored.sort(key=lambda pair: -pair[0])
    best = float(scored[0][0]) if scored else 0.0

    picked: list[dict] = []
    for similarity, chunk in scored:
        if len(picked) >= k or similarity < min_score:
            break
        item = dict(chunk)
        item["semantic_score"] = float(similarity)
        picked.append(item)
    return picked, best


def rerank(question: str, chunks: list[dict], top_n: int = 6) -> list[dict]:
    """Переупорядочить кандидатов реранкером; при сбое вернуть как было."""
    if not chunks or not rerank_available():
        return chunks[:top_n]
    documents = [
        f"{chunk.get('doc', '')}, п. {chunk.get('clause', '')}\n"
        f"{str(chunk.get('text', ''))[:SNIPPET_CHARS]}"
        for chunk in chunks
    ]
    try:
        result = _post("/rerank", {
            "model": RERANKER_MODEL,
            "query": question,
            "documents": documents,
            "top_n": min(top_n, len(chunks)),
        }, api_key("rerank"))
        items = result.get("results", result.get("data", []))
        picked: list[dict] = []
        seen: set[int] = set()
        for item in items:
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= index < len(chunks)) or index in seen:
                continue
            seen.add(index)
            chunk = dict(chunks[index])
            score = item.get("relevance_score", item.get("score"))
            if score is not None:
                try:
                    chunk["rerank_score"] = float(score)
                except (TypeError, ValueError):
                    pass
            picked.append(chunk)
        if picked:
            return picked[:top_n]
    except Exception as exc:
        _disable("rerank", exc)
    return chunks[:top_n]


def order_by_similarity(question: str, chunks: list[dict]) -> list[dict]:
    """Переставить кандидатов по косинусной близости к вопросу."""
    if not chunks or not embeddings_available():
        return chunks
    vectors = embed([question] + [_chunk_text(chunk) for chunk in chunks])
    if not vectors or len(vectors) != len(chunks) + 1:
        return chunks
    query, rest = vectors[0], vectors[1:]
    scored = []
    for vector, chunk in zip(rest, chunks):
        similarity = sum(a * b for a, b in zip(query, vector))
        item = dict(chunk)
        item["semantic_score"] = float(similarity)
        scored.append((float(similarity), item))
    scored.sort(key=lambda pair: -pair[0])
    return [chunk for _, chunk in scored]
