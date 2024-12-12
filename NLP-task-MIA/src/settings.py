# settings.py
from pydantic import BaseSettings


class Settings(BaseSettings):
    MAX_FEATURES: int = 20000
    MAXLEN: int = 100

    class Config:
        env_file = ".env"  # Optional


settings = Settings()  # Instantiate settings
