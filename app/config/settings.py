import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    s3_url: str = "http://localhost:9000"
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio123"
    s3_verify_tls: str =

    timeout: int = 30
    max_file_size: int = 50 * 1024 * 1024
    cache_ttl: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()

endpoint = os.environ.get("S3_URL", "http://localhost:9000")
access_key = os.environ.get("S3_ACCESS_KEY", "minio")
secret_key = os.environ.get("S3_SECRET_KEY", "minio123")
verify = os.environ.get("S3_VERIFY_TLS", "false").lower() == "true"