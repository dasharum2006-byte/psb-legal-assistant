# backend/app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем индекс один раз при старте сервера"""
    try:
        # Правильный импорт по инструкции ревьюера
        from app.rag import retriever as rag_retriever

        logger.info(f"Loading index from {settings.INDEX_DIR}...")
        rag_retriever.load_index_from_disk(settings.INDEX_DIR)

        # Кладем в state сам модуль (у него есть .search())
        app.state.retriever = rag_retriever
        app.state.index_error = None
        logger.info("Index loaded successfully")

    except Exception as e:
        app.state.retriever = None
        app.state.index_error = str(e)
        logger.error(f"Failed to load index: {e}")

    yield


app = FastAPI(title="PSB Legal Assistant API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if "," in settings.CORS_ORIGINS else [settings.CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(documents_router, prefix="/api", tags=["documents"])


@app.get("/api/health")
def health():
    retriever = getattr(app.state, "retriever", None)
    error_msg = getattr(app.state, "index_error", "Unknown state")

    if retriever is not None:
        return {"status": "ok", "index": "loaded", "version": "0.1.0"}

    return {
        "status": "degraded",
        "index": None,
        "error": str(error_msg),
        "hint": "Run 'python scripts/build_index.py' and restart server"
    }