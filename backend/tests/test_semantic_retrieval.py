"""Гибридный поиск бэкенда: эмбеддинги и реранк должны реально влиять.

Зачётный сабмит выигрывает за счёт того, что семантика добавляет пункты,
которых нет в top-k BM25, а реранк меняет порядок контекста. В бэкенде
раньше эти шаги были декоративными: кандидаты не добавлялись, а
`select_chunks` сортировал всё обратно по BM25-score.

Эти тесты фиксируют поведение без сети: платформенные функции подменяются.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import orchestrator, semantic  # noqa: E402


def chunk(doc: str, clause: str, text: str, score: float = 0.0,
          doc_id: str | None = None) -> dict:
    return {
        "doc": doc,
        "doc_id": doc_id or doc.lower(),
        "doc_type": "contract",
        "clause": clause,
        "text": text,
        "searchable": f"{doc} {clause} {text}",
        "score": score,
        "status": "active",
    }


class Retriever:
    def __init__(self, found: list[dict], pool: list[dict]) -> None:
        self.found = found
        self.pool = pool
        self.requests: list[str] = []

    def search(self, query: str, k: int = 40) -> list[dict]:
        self.requests.append(query)
        return [dict(item) for item in self.found]

    def all_chunks(self) -> list[dict]:
        return [dict(item) for item in self.pool]

    def next_chunk(self, item: dict, limit: int = 1) -> list[dict]:
        return []


class Client:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str, system=None, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_semantic_candidate_missing_from_bm25_reaches_the_model(
    monkeypatch,
) -> None:
    bm25 = chunk("ДОГОВОР АРЕНДЫ", "1.1", "Арендодатель передаёт помещение.", score=30.0)
    memo = chunk(
        "СЛУЖЕБНАЯ ЗАПИСКА № ЮО-7", "абз. 1",
        "Оригиналы договоров выдаются в течение 5 рабочих дней.", score=0.0,
        doc_id="юо-7",
    )
    retriever = Retriever([bm25], [bm25, memo])
    client = Client("Оригиналы выдаются в течение 5 рабочих дней (абз. 1).")

    monkeypatch.setattr(orchestrator, "USE_SEMANTIC", True)
    monkeypatch.setattr(orchestrator, "USE_RERANK", True)
    monkeypatch.setattr(semantic, "embeddings_available", lambda: True)
    monkeypatch.setattr(semantic, "rerank_available", lambda: True)
    monkeypatch.setattr(
        semantic, "semantic_candidates",
        lambda question, chunks, k, min_score: (
            [dict(item, semantic_score=0.88) for item in chunks if item["doc_id"] == "юо-7"],
            0.88,
        ),
    )
    monkeypatch.setattr(semantic, "order_by_similarity", lambda question, chunks: chunks)
    monkeypatch.setattr(
        semantic, "rerank",
        lambda question, chunks, top_n: (
            [item for item in chunks if item["clause"] == "абз. 1"]
            + [item for item in chunks if item["clause"] != "абз. 1"]
        )[:top_n],
    )

    result = orchestrator.ask("в какой срок выдают оригиналы?", [], retriever, client=client)

    assert result["refusal"] is False
    assert "абз. 1" in client.prompts[0], "семантический кандидат не попал в контекст"
    assert [source["clause"] for source in result["sources"]] == ["абз. 1"]


def test_rerank_works_even_when_embeddings_are_disabled(monkeypatch) -> None:
    weak = chunk("ДОГОВОР № 1", "1.1", "Общие положения договора.", score=100.0, doc_id="one")
    right = chunk("ДОГОВОР № 2", "2.2", "Оплата в течение 10 дней.", score=90.0, doc_id="two")
    retriever = Retriever([weak, right], [weak, right])
    client = Client("Оплата в течение 10 дней (п. 2.2).")

    monkeypatch.setattr(orchestrator, "USE_SEMANTIC", True)
    monkeypatch.setattr(orchestrator, "USE_RERANK", True)
    monkeypatch.setattr(semantic, "embeddings_available", lambda: False)
    monkeypatch.setattr(semantic, "rerank_available", lambda: True)
    monkeypatch.setattr(
        semantic, "rerank",
        lambda question, chunks, top_n: [chunks[1], chunks[0]][:top_n],
    )

    result = orchestrator.ask("когда оплата?", [], retriever, client=client)

    assert result["refusal"] is False
    assert [source["clause"] for source in result["sources"]] == ["2.2"]
    assert client.prompts[0].index("2.2") < client.prompts[0].index("1.1"), (
        "select_chunks отсортировал контекст обратно по BM25 и отменил реранк"
    )


def test_embeddings_and_rerank_have_independent_availability(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://example.test")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDER_API_KEY", "embed-key")
    monkeypatch.setenv("RERANKER_API_KEY", "rerank-key")
    monkeypatch.setattr(
        semantic, "_disabled_until",
        {"embeddings": time.monotonic() + 60, "rerank": 0.0},
    )

    assert semantic.embeddings_available() is False
    assert semantic.rerank_available() is True
    assert semantic.api_key("embeddings") == "embed-key"
    assert semantic.api_key("rerank") == "rerank-key"


def test_embed_batches_and_reuses_cache(monkeypatch) -> None:
    calls: list[tuple[str, int, str]] = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            pass

    def fake_post(path: str, payload: dict, key: str, client=None) -> dict:
        calls.append((path, len(payload["input"]), key))
        return {
            "data": [
                {"index": index, "embedding": [1.0, float(index)]}
                for index in range(len(payload["input"]))
            ]
        }

    monkeypatch.setattr(semantic, "embeddings_available", lambda: True)
    monkeypatch.setattr(semantic, "api_key", lambda endpoint="embeddings": "embed-key")
    monkeypatch.setattr(semantic, "_post", fake_post)
    monkeypatch.setattr(semantic, "_make_client", lambda: FakeClient())
    monkeypatch.setattr(semantic, "_CACHE", {})

    texts = [f"текст {index}" for index in range(semantic.EMBED_BATCH + 3)]
    first = semantic.embed(texts)
    assert first is not None and len(first) == len(texts)
    assert len(calls) == 2, "эмбеддинги должны уходить батчами"

    second = semantic.embed(texts)
    assert second == first
    assert len(calls) == 2, "повторный вызов должен брать векторы из кэша"

    # кэш не должен смешивать векторы разных моделей
    monkeypatch.setattr(semantic, "EMB_MODEL", "another-embedder")
    semantic.embed(texts[:1])
    assert len(calls) == 3
