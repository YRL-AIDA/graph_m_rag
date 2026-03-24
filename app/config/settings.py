from typing import Optional, List

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class S3Settings(BaseSettings):
    """S3/MinIO storage configuration."""
    S3_URL: str = Field(default="http://localhost:9000", description="S3 endpoint URL")
    S3_ACCESS_KEY: str = Field(default="minio", description="S3 access key")
    S3_SECRET_KEY: str = Field(default="minio123", description="S3 secret key")
    S3_VERIFY_TLS: bool = Field(default=False, description="Verify TLS certificates")
    S3_BUCKET_NAME: str = Field(default="pdf-processing", description="Default bucket name")

    # MinIO specific
    MINIO_ROOT_USER: str = Field(default="minioadmin", description="MinIO root user")
    MINIO_ROOT_PASSWORD: str = Field(default="minioadmin", description="MinIO root password")
    MINIO_ENDPOINT: str = Field(default="minio:9000", description="MinIO internal endpoint for Docker")
    MINIO_BUCKET: str = Field(default="pdf-processing", description="MinIO bucket name")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant host")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant HTTP port")
    QDRANT_GRPC_PORT: int = Field(default=6334, description="Qdrant gRPC port")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key")
    QDRANT_COLLECTION_NAME: str = Field(default="documents", description="Default collection name")

    # For qdrant service
    host: str = Field(default="0.0.0.0", description="Qdrant service host")
    port: int = Field(default=8000, description="Qdrant service port")
    api_key: str = Field(default="", description="Qdrant service API key")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class MinerUSettings(BaseSettings):
    """MinerU document processing service configuration."""
    MINERU_HOST: str = Field(default="http://localhost", description="MinerU service host")
    MINERU_PORT: int = Field(default=8001, description="MinerU service port")
    MINERU_TIMEOUT: int = Field(default=300, description="Request timeout in seconds")
    MINERU_MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, description="Max file size in bytes")
    MINERU_CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")
    MODELSCOPE_CACHE: str = Field(default="/app/models", description="ModelScope cache directory")

    # Processing defaults
    MINERU_BACKEND: str = Field(default="pipeline", description="Default backend: pipeline or vlm")
    MINERU_METHOD: str = Field(default="auto", description="Default method: auto, txt, ocr")
    MINERU_LANG: str = Field(default="ru", description="Default document language")
    MINERU_FORMULA_ENABLE: bool = Field(default=True, description="Enable formula processing")
    MINERU_TABLE_ENABLE: bool = Field(default=True, description="Enable table processing")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class EmbeddingSettings(BaseSettings):
    """Embedding service configuration."""
    EMBEDDING_BASE_URL: str = Field(default="http://192.168.19.127:10114/embedding", description="Embedding service URL")
    EMBEDDING_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")
    EMBEDDING_MODEL: str = Field(default="qwen3-emb", description="Embedding model name")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

class LLMSettings(BaseSettings):
    """LLM service configuration."""
    LLM_BASE_URL: str = Field(default="http://192.168.19.127:8888/v1", description="LLM service URL")
    LLM_API_KEY: str = Field(default="EMPTY", description="LLM API key")
    LLM_MODEL_NAME: str = Field(default="Qwen/Qwen3-VL-32B-Thinking", description="LLM model name")
    LLM_MAX_TOKENS: int = Field(default=2048, description="Max tokens for response")
    LLM_TEMPERATURE: float = Field(default=0.7, description="Temperature for generation")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

class AppSettings(BaseSettings):
    """Main application configuration."""
    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Application host")
    PORT: int = Field(default=8000, description="Application port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # File processing
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, description="Max upload file size in bytes")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")
    TEMP_DIR: str = Field(default="/tmp/pdf_processing", description="Temporary directory for files")

    # CORS
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class Settings(BaseSettings):
    """
    Centralized settings class that aggregates all service configurations.
    Use this class to access any configuration setting in the application.
    """
    # Nested settings objects
    s3: S3Settings = Field(default_factory=S3Settings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    mineru: MinerUSettings = Field(default_factory=MinerUSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    app: AppSettings = Field(default_factory=AppSettings)

    # Direct access aliases for backward compatibility
    # S3
    S3_URL: str = Field(default="http://localhost:9000")
    S3_ACCESS_KEY: str = Field(default="minio")
    S3_SECRET_KEY: str = Field(default="minio123")
    S3_VERIFY_TLS: bool = Field(default=False)
    S3_BUCKET_NAME: str = Field(default="pdf-processing")

    # Qdrant
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    QDRANT_API_KEY: Optional[str] = Field(default=None)
    QDRANT_COLLECTION_NAME: str = Field(default="documents")

    # MinerU
    MINERU_HOST: str = Field(default="http://localhost")
    MINERU_PORT: int = Field(default=8001)
    MINERU_TIMEOUT: int = Field(default=300)
    MODELSCOPE_CACHE: str = Field(default="/app/models")

    # Embedding
    EMBEDDING_BASE_URL: str = Field(default="http://192.168.19.127:10114/embedding")
    EMBEDDING_TIMEOUT: int = Field(default=30)

    # App
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024)
    CACHE_TTL: int = Field(default=3600)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def mineru_url(self) -> str:
        """Get full MinerU service URL."""
        return f"{self.mineru.MINERU_HOST.rstrip('/')}:{self.mineru.MINERU_PORT}"

    @property
    def qdrant_url(self) -> str:
        """Get full Qdrant service URL."""
        return f"{self.qdrant.QDRANT_HOST}:{self.qdrant.QDRANT_PORT}"

    @property
    def s3_endpoint_clean(self) -> str:
        """Get S3 endpoint without http:// or https:// prefix for Minio client."""
        endpoint = self.s3.S3_URL
        if endpoint.startswith("http://"):
            return endpoint[7:]
        elif endpoint.startswith("https://"):
            return endpoint[8:]
        return endpoint

    @property
    def s3_secure(self) -> bool:
        """Check if S3 connection should use HTTPS."""
        return self.s3.S3_URL.startswith("https://")


# Create global settings instance
settings = Settings()


# Backward compatibility - expose settings at module level
# This allows existing imports to continue working
S3_URL = settings.S3_URL
S3_ACCESS_KEY = settings.S3_ACCESS_KEY
S3_SECRET_KEY = settings.S3_SECRET_KEY
S3_VERIFY_TLS = settings.S3_VERIFY_TLS
S3_BUCKET_NAME = settings.S3_BUCKET_NAME

QDRANT_HOST = settings.QDRANT_HOST
QDRANT_PORT = settings.QDRANT_PORT
QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME

MINERU_HOST = settings.MINERU_HOST
MINERU_PORT = settings.MINERU_PORT