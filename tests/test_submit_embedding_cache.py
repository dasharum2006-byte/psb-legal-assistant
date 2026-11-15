"""Эмбеддинги чанков не должны считаться 6 раз параллельно.

В solution.py вопросы обрабатывает ThreadPoolExecutor(max_workers=6).
Без прогрева и блокировки первые 6 потоков видели пустой кэш и отправляли
в /embeddings одни и те же 324 чанка.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import submit.solution as solution  # noqa: E402


def test_warm_embeddings_prefills_searchable_chunks(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_embed(client, texts):
        seen.append(list(texts))
        return [[0.0] for _ in texts]

    monkeypatch.setattr(solution, "_embed", fake_embed)
    chunks = [{"searchable": "x" * 2000}, {"searchable": "y"}]

    assert solution.warm_embeddings(object(), chunks) == 2
    assert seen == [["x" * 1200, "y"]]
    assert solution.warm_embeddings(None, chunks) == 0


def test_embed_deduplicates_concurrent_calls(monkeypatch) -> None:
    monkeypatch.setattr(solution, "_EMBEDDING_CACHE", {})
    calls: list[list[str]] = []

    def fake_post(client, path, key, payload):
        calls.append(list(payload["input"]))
        time.sleep(0.02)          # даём остальным потокам дойти до _embed
        return {
            "data": [
                {"index": index, "embedding": [1.0, 0.0]}
                for index in range(len(payload["input"]))
            ]
        }

    monkeypatch.setattr(solution, "_post", fake_post)
    texts = ["alpha", "beta"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: solution._embed(None, texts), range(6)))

    assert all(result == results[0] for result in results)
    assert calls == [["alpha", "beta"]], (
        f"вместо одного запроса было {len(calls)}: блокировка кэша не работает"
    )


def test_reorder_by_embeddings_uses_shared_embed(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_embed(client, texts):
        calls.append(list(texts))
        return [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]

    monkeypatch.setattr(solution, "_embed", fake_embed)
    chunks = [{"text": "далёкий текст"}, {"text": "близкий текст"}]

    ordered = solution.reorder_by_embeddings(None, "вопрос", chunks)

    assert calls == [["вопрос", "далёкий текст", "близкий текст"]]
    assert [chunk["text"] for chunk in ordered] == ["близкий текст", "далёкий текст"]
