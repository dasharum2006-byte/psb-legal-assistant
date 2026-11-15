"""Кэш эмбеддингов бэкенда: прогрев и защита от параллельного дублирования."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import semantic  # noqa: E402


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_warm_embeddings_prefills_chunk_cache(monkeypatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(semantic, "embeddings_available", lambda: True)
    monkeypatch.setattr(
        semantic, "embed",
        lambda texts: seen.append(list(texts)) or [[0.0] for _ in texts],
    )
    chunks = [{"searchable": "первый"}, {"searchable": "второй"}]

    assert semantic.warm_embeddings(chunks) == 2
    assert seen == [["первый", "второй"]]


def test_embed_does_not_duplicate_parallel_requests(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(semantic, "_CACHE", {})
    monkeypatch.setattr(semantic, "_disabled_until", {})
    monkeypatch.setattr(semantic, "embeddings_available", lambda: True)
    monkeypatch.setattr(semantic, "api_key", lambda endpoint="embeddings": "test-key")
    monkeypatch.setattr(semantic, "_make_client", lambda: FakeClient())

    def fake_post(path, payload, key, client=None):
        calls.append(list(payload["input"]))
        time.sleep(0.02)
        return {
            "data": [
                {"index": index, "embedding": [1.0, 0.0]}
                for index in range(len(payload["input"]))
            ]
        }

    monkeypatch.setattr(semantic, "_post", fake_post)
    texts = ["alpha", "beta"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: semantic.embed(texts), range(6)))

    assert all(result == results[0] for result in results)
    assert calls == [["alpha", "beta"]]
