from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent


class DatabaseSettings(BaseSettings):
    DB_HOST: str = None
    DB_PORT: int = None
    DB_USER: str = None
    DB_PASSWORD: str = None
    DB_NAME: str = None

    @property
    def async_url(self) -> str:
        return f"postgresql+psycopg_async://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def sync_url(self) -> str:
        return f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    echo: bool = True
    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env")


class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()


settings = Settings()
