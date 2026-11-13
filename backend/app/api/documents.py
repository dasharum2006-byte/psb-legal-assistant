# backend/app/api/documents.py
from fastapi import APIRouter, Depends
from app.api.deps import get_retriever

router = APIRouter(tags=["documents"])


@router.get("/documents")
async def get_documents(retriever=Depends(get_retriever)):
    """
    Возвращает список документов из индекса.
    В тестах используется MockRetriever через dependency_overrides.
    """
    # Безопасное получение списка документов из мок-ретривера
    docs = getattr(retriever, "documents", [])

    return {"documents": docs}