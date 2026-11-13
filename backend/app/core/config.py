from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TOP_K: int = 8
    MIN_SCORE: float = 0.25
    REQUEST_TIMEOUT: int = 30
    INDEX_DIR: str = "./data/index"
    CORS_ORIGINS: str = "*"

    class Config:
        env_file = ".env"


settings = Settings()