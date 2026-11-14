"""Ответ не должен падать в 500 из-за «грязных» источников.

Оркестратор может вернуть None в clause (документы без нумерации),
numpy.float32 или None в score. Раньше это давало ValidationError -> 500.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import chat as chat_module  # noqa: E402
from app.api.chat import ChatResponse, sanitize_result  # noqa: E402
from app.api.deps import get_retriever  # noqa: E402
from app.main import app  # noqa: E402


class FakeRetriever:
    def search(self, query, k=8, filters=None):
        return []


def test_none_values_do_not_break_response() -> None:
    dirty = {
        "answer": None,
        "sources": [
            {"doc": None, "clause": None, "text": None, "score": None},
            {"doc": "Договор", "clause": 31, "text": "текст", "score": np.float32(0.83)},
        ],
        "refusal": None,
        "missing_info": None,
    }
    response = ChatResponse(**sanitize_result(dirty))
    assert response.answer
    assert response.sources[0].clause == ""
    assert response.sources[1].clause == "31"
    assert isinstance(response.sources[1].score, float)


def test_api_returns_200_when_orchestrator_gives_dirty_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def dirty_ask(question, history, retriever, client=None, filters=None):
        return {"answer": "Ответ", "sources": [{"doc": None, "clause": None, "text": None, "score": None}],
                "refusal": False, "missing_info": []}

    monkeypatch.setattr(chat_module, "load_ask", lambda: dirty_ask)
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()

    response = TestClient(app).post("/api/chat", json={"message": "вопрос", "history": [], "filters": {}})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Ответ"
    assert body["sources"][0]["clause"] == ""
    app.dependency_overrides.clear()


def test_numpy_score_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    def numpy_ask(question, history, retriever, client=None, filters=None):
        return {"answer": "Ответ", "sources": [{"doc": "Д", "clause": "1.1", "text": "т",
                                                "score": np.float32(0.5)}],
                "refusal": False, "missing_info": []}

    monkeypatch.setattr(chat_module, "load_ask", lambda: numpy_ask)
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()

    response = TestClient(app).post("/api/chat", json={"message": "вопрос", "history": [], "filters": {}})

    assert response.status_code == 200
    assert response.json()["sources"][0]["score"] == 0.5
    app.dependency_overrides.clear()
