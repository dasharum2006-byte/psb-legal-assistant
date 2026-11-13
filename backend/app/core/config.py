# backend/app/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TOP_K: int = 8
    MIN_SCORE: float = 12.0  # Дефолт для BM25, как просил ревьюер
    REQUEST_TIMEOUT: int = 30
    INDEX_DIR: str = "./data/index"
    CORS_ORIGINS: str = "*"

    class Config:
        env_file = ".env"


settings = Settings()