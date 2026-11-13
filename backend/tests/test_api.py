# backend/tests/test_api.py
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from app.main import app
from fastapi import Request
from app.api.deps import get_retriever


class MockRetriever:
    """Фиктивный ретривер для тестов"""

    def __init__(self):
        self.name = "test-index"
        # Имитируем _chunks_store для эндпоинта /documents
        self._chunks_store = [
            {
                "doc_id": "doc1",
                "title": "Договор аренды №А-08/2024",
                "type": "contract",
                "date": "2024-01-01",
                "status": "active"
            }
        ]
        # Для совместимости с другими частями кода
        self.documents = self._chunks_store


def mock_get_retriever(request: Request):
    return MockRetriever()


app.dependency_overrides[get_retriever] = mock_get_retriever

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]


def test_chat_empty_message():
    response = client.post(
        "/api/chat",
        json={"message": "", "history": [], "filters": {}}
    )
    assert response.status_code == 422


def test_documents():
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)
    assert len(data["documents"]) > 0

    # Проверяем наличие обязательных полей
    first_doc = data["documents"][0]
    assert "title" in first_doc
    assert "doc_id" in first_doc


@pytest.fixture(autouse=True)
def cleanup_overrides():
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides[get_retriever] = mock_get_retriever