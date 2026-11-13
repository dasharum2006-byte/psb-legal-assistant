"""Список документов базы знаний для фронта."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_retriever

router = APIRouter(tags=["documents"])


def _documents_from_chunks(chunks: list[dict]) -> list[dict]:
    """Собрать документы из чанков: один документ — одна запись."""
    order: list[str] = []
    counters: dict[str, int] = {}
    first: dict[str, dict] = {}

    for chunk in chunks:
        key = chunk.get("doc_id") or chunk.get("doc") or chunk.get("title")
        if not key:
            continue
        counters[key] = counters.get(key, 0) + 1
        if key not in first:
            order.append(key)
            first[key] = chunk

    documents = []
    for key in order:
        chunk = first[key]
        documents.append({
            "doc_id": key,
            "title": chunk.get("doc") or chunk.get("title") or key,
            "type": chunk.get("doc_type"),
            "number": chunk.get("number"),
            "date": chunk.get("valid_from") or chunk.get("date"),
            "status": chunk.get("status"),
            "revision": chunk.get("revision"),
            "chunks": counters[key],
        })

    documents.sort(key=lambda item: (item["status"] != "active", item["title"]))
    return documents


@router.get("/documents")
async def get_documents(retriever=Depends(get_retriever)):
    """Список документов из загруженного индекса."""
    # 1) индекс настоящий: чанки лежат в модуле retriever
    chunks = getattr(retriever, "_chunks_store", None)
    if chunks:
        return {"documents": _documents_from_chunks(chunks)}

    # 2) тестовый мок: у него готовый список
    predefined = getattr(retriever, "documents", None)
    if predefined:
        return {"documents": predefined}

    return {"documents": []}
