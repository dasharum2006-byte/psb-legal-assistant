from fastapi import APIRouter

router = APIRouter(tags=["documents"])


@router.get("/documents")
def list_documents() -> dict:
    """Заглушка списка документов. Позже здесь будет база знаний."""
    return {"documents": []}
