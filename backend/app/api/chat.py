from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict)


class Source(BaseModel):
    doc: str
    clause: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    refusal: bool = False
    missing_info: list[str] = Field(default_factory=list)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Заглушка. Реальный RAG будет подключён на следующем шаге."""
    return ChatResponse(
        answer="RAG ещё не подключён. Это заглушка API.",
        sources=[],
        refusal=True,
        missing_info=["нужно подключить поиск по базе знаний"],
    )
