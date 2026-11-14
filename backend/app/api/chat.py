# backend/app/api/chat.py
import asyncio
import logging
import uuid
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.api.deps import get_retriever
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


# --- СХЕМЫ (сохраняем контракт с фронтом) ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Текст вопроса")
    history: list[dict] = Field(default_factory=list, description="История диалога")
    filters: dict = Field(default_factory=dict, description="Фильтры поиска")

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("history", mode="after")
    @classmethod
    def limit_history(cls, v: list[dict]) -> list[dict]:
        return v[-20:] if len(v) > 20 else v


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


# --- ЭНДПОИНТ ---

def load_ask():
    """Импорт оркестратора отдельной функцией.

    Раньше здесь стоял `except ImportError` с откатом на заглушку: при любой
    проблеме с импортом (нет библиотеки в контейнере, кривой путь) API молча
    отдавал фальшивый ответ. Теперь ошибка видна и возвращается 503.
    """
    from app.rag.orchestrator import ask

    return ask


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
        request: ChatRequest,
        retriever=Depends(get_retriever),
):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        ask = load_ask()

        result = await asyncio.wait_for(
            asyncio.to_thread(
                ask,
                request.message,
                request.history,
                retriever,
                None,                    # client=LLMClient() по умолчанию
                request.filters,         # фильтры из запроса больше не теряются
            ),
            timeout=settings.REQUEST_TIMEOUT
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"CHAT [{request_id}] success | "
            f"latency={latency_ms}ms | "
            f"refusal={result.get('refusal')} | "
            f"sources_count={len(result.get('sources', []))}"
        )

        return ChatResponse(**result)

    except ImportError:
        # Никаких заглушек: если оркестратор не импортируется — это авария,
        # её надо видеть, а не маскировать фальшивым ответом.
        logger.exception(f"CHAT [{request_id}] ORCHESTRATOR_IMPORT_FAILED")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Оркестратор недоступен: ошибка импорта",
                "hint": "проверьте зависимости и импорты в app.rag.orchestrator",
            },
        )

    except asyncio.TimeoutError:
        logger.error(f"CHAT [{request_id}] TIMEOUT after {settings.REQUEST_TIMEOUT}s")
        raise HTTPException(status_code=504, detail="Пайплайн не ответил вовремя")

    except ValueError as e:
        logger.warning(f"CHAT [{request_id}] BAD_REQUEST: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"CHAT [{request_id}] INTERNAL_ERROR")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")