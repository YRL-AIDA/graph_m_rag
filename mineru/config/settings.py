from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "http://localhost"
    port: str = "8000"
    timeout: int = 30
    max_file_size: int = 50 * 1024 * 1024
    cache_ttl: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()
