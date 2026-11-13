# backend/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
from fastapi import Request
from app.api.deps import get_retriever


class MockRetriever:
    """Фиктивный ретривер для тестов"""
    def __init__(self):
        self.name = "test-index"
        # Структура должна точно совпадать с тем, что ждет фронт и эндпоинт
        self.documents = [
            {
                "doc_id": "doc1",
                "title": "Test Contract",
                "type": "contract",
                "number": "123",
                "date": "2024-01-01"
            }
        ]


def mock_get_retriever(request: Request):
    return MockRetriever()


# ⚠️ КРИТИЧЕСКИ ВАЖНО: Оверрайд ДО создания клиента
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


@pytest.fixture(autouse=True)
def cleanup_overrides():
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides[get_retriever] = mock_get_retriever