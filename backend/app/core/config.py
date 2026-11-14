# backend/app/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TOP_K: int = 8
    MIN_SCORE: float = 8.0  # шкала BM25 (0..1 — для гибрида)
    # должно быть БОЛЬШЕ, чем худший случай клиента LLM:
    # LLM_TIMEOUT * LLM_RETRIES + паузы. Иначе API отдаёт 504 раньше,
    # чем клиент успевает повторить запрос.
    REQUEST_TIMEOUT: int = 60
    INDEX_DIR: str = "./data/index"
    CORS_ORIGINS: str = "*"

    class Config:
        env_file = ".env"


settings = Settings()