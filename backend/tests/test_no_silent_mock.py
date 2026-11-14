"""Никаких тихих заглушек: ошибка импорта оркестратора должна быть видна.

Раньше `except ImportError` в api/chat.py откатывался на mock_ask и отдавал
фальшивый ответ с фейковым источником — на проверке это нули за ответы.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import chat as chat_module  # noqa: E402
from app.api.deps import get_retriever  # noqa: E402
from app.main import app  # noqa: E402


class FakeRetriever:
    def search(self, query, k=8, filters=None):
        return [{
            "doc_id": "doc", "doc": "ДОГОВОР № 1", "doc_type": "contract",
            "clause": "1.1", "text": "Текст условия.", "score": 30.0, "status": "active",
        }]


def _client_with_retriever() -> TestClient:
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    return TestClient(app)


def test_import_error_returns_503_not_fake_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_load_ask():
        raise ImportError("нет библиотеки rank_bm25")

    monkeypatch.setattr(chat_module, "load_ask", broken_load_ask)

    response = _client_with_retriever().post(
        "/api/chat",
        json={"message": "какая арендная плата?", "history": [], "filters": {}},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "Оркестратор недоступен" in detail["message"]
    # главное: фальшивого ответа нет
    assert "Тестовый ответ" not in response.text
    app.dependency_overrides.clear()


def test_no_mock_ask_left_in_module() -> None:
    assert not hasattr(chat_module, "mock_ask"), "заглушка ответов должна быть удалена"


def test_health_reports_orchestrator_state() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert "orchestrator" in body
