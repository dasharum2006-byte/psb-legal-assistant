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
    doc: str = ""
    clause: str = ""
    text: str = ""
    score: float = 0.0

    @field_validator("doc", "clause", "text", mode="before")
    @classmethod
    def coerce_text(cls, value):
        """None и числа приводим к строке: иначе валидация роняет весь ответ."""
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score(cls, value):
        """numpy.float32, Decimal, строка, None — всё сводим к float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def _as_text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sanitize_result(result: dict | None) -> dict:
    """Привести словарь оркестратора к видам, которые ждёт схема ответа.

    Оркестратор может вернуть None в clause (документы без нумерации),
    numpy.float32 в score или пустой answer — всё это раньше роняло ответ
    в 500 через ошибку валидации.
    """
    result = result or {}
    sources = []
    for item in result.get("sources") or []:
        item = item or {}
        sources.append({
            "doc": _as_text(item.get("doc")),
            "clause": _as_text(item.get("clause")),
            "text": _as_text(item.get("text")),
            "score": _as_float(item.get("score")),
        })

    return {
        "answer": _as_text(result.get("answer")) or "Не удалось сформировать ответ.",
        "sources": sources,
        "refusal": bool(result.get("refusal")),
        "missing_info": [_as_text(item) for item in (result.get("missing_info") or [])],
    }


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

        return ChatResponse(**sanitize_result(result))

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