from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TOP_K: int = 8
    MIN_SCORE: float = 8.0  # шкала BM25 (0..1 — для гибрида)
    REQUEST_TIMEOUT: int = 30
    INDEX_DIR: str = "./data/index"
    CORS_ORIGINS: str = "*"

    class Config:
        env_file = ".env"


settings = Settings()