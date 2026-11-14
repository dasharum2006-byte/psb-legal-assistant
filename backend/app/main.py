# backend/app/main.py
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
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
    # Проверяем импорт оркестратора сразу: если он сломан (нет зависимости,
    # кривой путь), лучше видеть это на старте и в /api/health, чем ловить
    # 503 на каждом вопросе.
    try:
        from app.api.chat import load_ask

        load_ask()
        app.state.orchestrator_error = None
        logger.info("Orchestrator imported successfully")
    except Exception as exc:                      # noqa: BLE001
        app.state.orchestrator_error = f"{type(exc).__name__}: {exc}"
        logger.exception("ORCHESTRATOR IMPORT FAILED")

    try:
        # Правильный импорт по инструкции ревьюера
        from app.rag import retriever as rag_retriever

        # «./data/index» считаем от корня проекта, а не от папки запуска:
        # иначе при запуске из backend/ индекс не находится
        index_dir = Path(settings.INDEX_DIR)
        if not index_dir.is_absolute():
            index_dir = Path(__file__).resolve().parents[2] / str(settings.INDEX_DIR).lstrip("./")

        logger.info(f"Loading index from {index_dir}...")
        rag_retriever.load_index_from_disk(str(index_dir))
        if not getattr(rag_retriever, "_chunks_store", None):
            raise RuntimeError("индекс пуст: сначала запустите scripts/build_index.py")

        # Проверяем, что индекс собран тем же кодом нарезки
        app.state.index_stale = None
        meta_path = index_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # считаем от корня проекта, как это делает scripts/build_index.py
            repo_root = Path(__file__).resolve().parents[2]
            fingerprint = ""
            for name in ("chunker.py", "parser.py"):
                code = repo_root / "backend/app/rag" / name
                if code.exists():
                    fingerprint += hashlib.sha256(code.read_bytes()).hexdigest()
            actual = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
            if meta.get("code_fingerprint") != actual:
                app.state.index_stale = (
                    f"индекс собран {meta.get('built_at')} другим кодом нарезки — "
                    f"пересоберите: python scripts/build_index.py"
                )
                logger.warning("INDEX_STALE: %s", app.state.index_stale)
        else:
            app.state.index_stale = "нет meta.json — пересоберите индекс: python scripts/build_index.py"
            logger.warning("INDEX_STALE: %s", app.state.index_stale)

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

    orchestrator = getattr(app.state, "orchestrator_error", None) or "ok"

    if retriever is not None:
        return {
            "status": "ok" if orchestrator == "ok" else "degraded",
            "index": "loaded",
            "orchestrator": orchestrator,
            "index_stale": getattr(app.state, "index_stale", None),
            "version": "0.1.0",
        }

    return {
        "status": "degraded",
        "index": None,
        "orchestrator": orchestrator,
        "error": str(error_msg),
        "hint": "Run 'python scripts/build_index.py' and restart server"
    }