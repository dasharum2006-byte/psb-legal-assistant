# backend/app/api/documents.py
import re
from fastapi import APIRouter, Depends, Query
from typing import Optional

from app.api.deps import get_retriever

router = APIRouter(tags=["documents"])


@router.get("/documents")
async def get_documents(
        retriever=Depends(get_retriever),
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0)
):
    """
    Возвращает список уникальных документов из индекса.
    Собирается из _chunks_store модуля ретривера.
    """
    unique_docs = {}

    # Проверяем наличие хранилища чанков
    chunks_store = getattr(retriever, "_chunks_store", [])

    for chunk in chunks_store:
        doc_id = chunk.get("doc_id") or chunk.get("id")
        if not doc_id or doc_id in unique_docs:
            continue

        title = chunk.get("doc") or chunk.get("title") or "Без названия"

        # Извлекаем номер из заголовка регуляркой, если нет отдельного поля
        number_match = re.search(r'№\s*[А-ЯA-Z0-9./-]+', title)
        number = number_match.group(0).strip() if number_match else ""

        unique_docs[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            # в нашем индексе поля называются doc_type / valid_from / revision
            "type": chunk.get("doc_type") or chunk.get("type") or "unknown",
            "number": chunk.get("number") or number,
            "date": chunk.get("valid_from") or chunk.get("date") or "",
            "status": chunk.get("status") or "active",
            "revision": chunk.get("revision"),
        }

    docs = list(unique_docs.values())

    # Фильтрация по статусу
    if status:
        docs = [d for d in docs if d.get("status") == status]

    # Сортировка: сначала active, потом по названию
    def sort_key(doc):
        is_active = 0 if doc.get("status") == "active" else 1
        return (is_active, doc.get("title", "").lower())

    docs.sort(key=sort_key)

    # Пагинация
    total = len(docs)
    paginated_docs = docs[offset: offset + limit]

    return {
        "documents": paginated_docs,
        "total": total,
        "limit": limit,
        "offset": offset
    }