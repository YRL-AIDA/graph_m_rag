# config/settings.py
import os
import secrets
from typing import List, Optional, Dict, Any, Union
from pydantic import AnyHttpUrl, BaseModel, Field, validator, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings
from enum import Enum
from functools import lru_cache


class Environment(str, Enum):
    """Типы окружений"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Уровни логирования"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Типы баз данных"""
    POSTGRES = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"


# Модели для вложенных конфигураций
class MinioConfig(BaseModel):
    """Конфигурация MinIO"""
    endpoint: str = "minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "rag-documents"
    secure: bool = False
    region: Optional[str] = None

    class Config:
        env_prefix = "MINIO_"


class QdrantConfig(BaseModel):
    """Конфигурация Qdrant"""
    url: str = "http://qdrant:6333"
    api_key: Optional[str] = None
    timeout: int = 30
    collection_name: str = "documents"
    vector_size: int = 768
    distance: str = "Cosine"  # Cosine, Euclidean, Dot

    class Config:
        env_prefix = "QDRANT_"


class Neo4jConfig(BaseModel):
    """Конфигурация Neo4j"""
    uri: str = "bolt://neo4j:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50

    class Config:
        env_prefix = "NEO4J_"


class OllamaConfig(BaseModel):
    """Конфигурация Ollama (Qwen модели)"""
    url: str = "http://ollama:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:7b-instruct"
    vl_model: str = "qwen2.5-vl:7b"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 300

    class Config:
        env_prefix = "OLLAMA_"


class MinerUConfig(BaseModel):
    """Конфигурация MinerU"""
    url: str = "http://127.0.0.1:8000"
    timeout: int = 300
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    models_cache_dir: str = "/app/models"

    class Config:
        env_prefix = "MINERU_"


class RedisConfig(BaseModel):
    """Конфигурация Redis"""
    url: RedisDsn = "redis://redis:6379/0"
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5

    class Config:
        env_prefix = "REDIS_"


class PostgresConfig(BaseModel):
    """Конфигурация PostgreSQL"""
    host: str = "postgres"
    port: int = 5432
    user: str = "raguser"
    password: str = "password"
    database: str = "ragdb"
    pool_size: int = 20
    echo: bool = False

    @property
    def dsn(self) -> str:
        """Получить DSN строку для подключения"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_prefix = "POSTGRES_"


class SecurityConfig(BaseModel):
    """Конфигурация безопасности"""
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = []
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

    class Config:
        env_prefix = "SECURITY_"


class LoggingConfig(BaseModel):
    """Конфигурация логирования"""
    level: LogLevel = LogLevel.INFO
    format: str = "json"  # json, text
    file_path: Optional[str] = "/app/logs/app.log"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True

    class Config:
        env_prefix = "LOGGING_"


class RAGConfig(BaseModel):
    """Конфигурация RAG системы"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_document: int = 1000
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    top_k_results: int = 5
    top_k_rerank: int = 3
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    enable_summarization: bool = True

    class Config:
        env_prefix = "RAG_"


class EmbeddingConfig(BaseModel):
    """Конфигурация эмбеддингов"""
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    pooling_method: str = "mean"  # mean, cls, max
    device: str = "cpu"  # cpu, cuda, mps

    class Config:
        env_prefix = "EMBEDDING_"


class LLMConfig(BaseModel):
    """Конфигурация LLM"""
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = []
    stream: bool = False

    class Config:
        env_prefix = "LLM_"


class FileProcessingConfig(BaseModel):
    """Конфигурация обработки файлов"""
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = [
        ".pdf", ".txt", ".md", ".doc", ".docx",
        ".jpg", ".jpeg", ".png", ".tiff", ".bmp"
    ]
    temp_dir: str = "/tmp/rag_uploads"
    keep_temp_files: bool = False
    max_concurrent_uploads: int = 10

    class Config:
        env_prefix = "FILE_"


class MonitoringConfig(BaseModel):
    """Конфигурация мониторинга"""
    enabled: bool = False
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    health_check_path: str = "/health"
    enable_tracing: bool = False
    enable_profiling: bool = False

    class Config:
        env_prefix = "MONITORING_"


class APIConfig(BaseModel):
    """Конфигурация API"""
    title: str = "RAG Service API"
    description: str = "RAG Service with MinIO, Qdrant, Neo4j, MinerU and Qwen models"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1

    class Config:
        env_prefix = "API_"


class Settings(BaseSettings):
    """Основные настройки приложения"""

    # Основные настройки
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    project_name: str = "rag-service"
    version: str = "1.0.0"

    # Конфигурации сервисов
    minio: MinioConfig = MinioConfig()
    qdrant: QdrantConfig = QdrantConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    ollama: OllamaConfig = OllamaConfig()
    mineru: MinerUConfig = MinerUConfig()
    redis: Optional[RedisConfig] = None
    postgres: Optional[PostgresConfig] = None

    # Функциональные конфигурации
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    rag: RAGConfig = RAGConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    file_processing: FileProcessingConfig = FileProcessingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    api: APIConfig = APIConfig()

    # Пути и директории
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = "/app/data"
    models_dir: str = "/app/models"
    logs_dir: str = "/app/logs"

    # Временные переменные
    test_mode: bool = False
    skip_initialization: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False

    @property
    def is_development(self) -> bool:
        """Проверка на development окружение"""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Проверка на production окружение"""
        return self.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Проверка на testing окружение"""
        return self.environment == Environment.TESTING or self.test_mode

    def get_database_url(self, db_type: DatabaseType = DatabaseType.POSTGRES) -> Optional[str]:
        """Получение URL базы данных в зависимости от типа"""
        if db_type == DatabaseType.POSTGRES and self.postgres:
            return self.postgres.dsn
        elif db_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.data_dir}/rag.db"
        return None

    def get_cache_url(self) -> Optional[str]:
        """Получение URL кэша"""
        if self.redis:
            return str(self.redis.url)
        return None

    def validate_settings(self) -> List[str]:
        """Валидация настроек и возврат списка ошибок"""
        errors = []

        # Проверка необходимых конфигураций
        if not self.minio.endpoint:
            errors.append("MinIO endpoint не указан")

        if not self.qdrant.url:
            errors.append("Qdrant URL не указан")

        if not self.neo4j.uri:
            errors.append("Neo4j URI не указан")

        if not self.ollama.url:
            errors.append("Ollama URL не указан")

        if not self.mineru.url:
            errors.append("MinerU URL не указан")

        # Проверка портов
        if self.api.port < 1 or self.api.port > 65535:
            errors.append(f"Неверный порт API: {self.api.port}")

        if self.monitoring.prometheus_port < 1 or self.monitoring.prometheus_port > 65535:
            errors.append(f"Неверный порт Prometheus: {self.monitoring.prometheus_port}")

        # Проверка директорий
        required_dirs = [self.data_dir, self.models_dir, self.logs_dir]
        for directory in required_dirs:
            if not os.path.isabs(directory):
                errors.append(f"Директория должна быть абсолютным путем: {directory}")

        return errors

    def get_service_urls(self) -> Dict[str, str]:
        """Получение всех URL сервисов"""
        return {
            "minio": self.minio.endpoint,
            "qdrant": self.qdrant.url,
            "neo4j": self.neo4j.uri,
            "ollama": self.ollama.url,
            "mineru": self.mineru.url,
            "api": f"http://{self.api.host}:{self.api.port}",
        }

    def get_health_check_urls(self) -> Dict[str, str]:
        """Получение URL для проверки здоровья сервисов"""
        return {
            "minio": f"{self.minio.endpoint}/minio/health/live",
            "qdrant": f"{self.qdrant.url}/health",
            "neo4j": f"http://{self.neo4j.uri.replace('bolt://', '').split(':')[0]}:7474",
            "ollama": f"{self.ollama.url}/api/tags",
            "mineru": f"{self.mineru.url}/health",
            "api": f"http://{self.api.host}:{self.api.port}/health",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование настроек в словарь (без секретов)"""
        config_dict = self.dict()

        # Маскируем секретные данные
        if config_dict.get("minio"):
            config_dict["minio"]["secret_key"] = "***"

        if config_dict.get("neo4j"):
            config_dict["neo4j"]["password"] = "***"

        if config_dict.get("security"):
            config_dict["security"]["secret_key"] = "***"
            config_dict["security"]["api_keys"] = ["***" for _ in config_dict["security"]["api_keys"]]

        if config_dict.get("redis") and config_dict["redis"]:
            config_dict["redis"]["password"] = "***"

        if config_dict.get("postgres") and config_dict["postgres"]:
            config_dict["postgres"]["password"] = "***"

        return config_dict


@lru_cache()
def get_settings() -> Settings:
    """
    Получение экземпляра настроек с кэшированием.
    Это стандартный способ использования настроек в FastAPI.
    """
    settings = Settings()

    # Валидация настроек
    errors = settings.validate_settings()
    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(f"Ошибки конфигурации:\n{error_msg}")

    # Создание необходимых директорий в development окружении
    if settings.is_development and not settings.skip_initialization:
        os.makedirs(settings.data_dir, exist_ok=True)
        os.makedirs(settings.models_dir, exist_ok=True)
        os.makedirs(settings.logs_dir, exist_ok=True)
        os.makedirs(settings.file_processing.temp_dir, exist_ok=True)

    return settings


# Создаем глобальный экземпляр настроек для импорта
try:
    settings = get_settings()
except Exception as e:
    # Fallback на значения по умолчанию при ошибках загрузки
    print(f"Warning: Failed to load settings: {e}")
    settings = Settings()


# Конфигурация для тестирования
class TestSettings(Settings):
    """Настройки для тестового окружения"""

    class Config:
        env_file = ".env.test"
        env_file_encoding = "utf-8"

    environment: Environment = Environment.TESTING
    test_mode: bool = True
    skip_initialization: bool = True

    # Переопределяем настройки для тестов
    minio: MinioConfig = MinioConfig(
        endpoint="localhost:9001",
        bucket="test-rag-documents"
    )

    qdrant: QdrantConfig = QdrantConfig(
        url="http://localhost:6334",
        collection_name="test-documents"
    )

    neo4j: Neo4jConfig = Neo4jConfig(
        uri="bolt://localhost:7688",
        database="test-neo4j"
    )

    ollama: OllamaConfig = OllamaConfig(
        url="http://localhost:11435"
    )

    mineru: MinerUConfig = MinerUConfig(
        url="http://localhost:8002"
    )

    # Используем SQLite для тестов
    postgres: Optional[PostgresConfig] = None

    # Отключаем ненужные функции в тестах
    monitoring: MonitoringConfig = MonitoringConfig(enabled=False)
    logging: LoggingConfig = LoggingConfig(level=LogLevel.WARNING)


def get_test_settings() -> TestSettings:
    """Получение настроек для тестирования"""
    return TestSettings()


# Утилитарные функции для работы с настройками
def get_config_summary() -> Dict[str, Any]:
    """Получение сводки конфигурации"""
    return {
        "environment": settings.environment.value,
        "debug": settings.debug,
        "project_name": settings.project_name,
        "version": settings.version,
        "services": {
            "minio": {"endpoint": settings.minio.endpoint, "bucket": settings.minio.bucket},
            "qdrant": {"url": settings.qdrant.url, "collection": settings.qdrant.collection_name},
            "neo4j": {"uri": settings.neo4j.uri, "database": settings.neo4j.database},
            "ollama": {"url": settings.ollama.url, "models": {
                "embedding": settings.ollama.embedding_model,
                "llm": settings.ollama.llm_model,
                "vl": settings.ollama.vl_model
            }},
            "mineru": {"url": settings.mineru.url},
        },
        "api": {
            "host": settings.api.host,
            "port": settings.api.port,
            "docs_url": settings.api.docs_url
        },
        "rag": {
            "chunk_size": settings.rag.chunk_size,
            "chunk_overlap": settings.rag.chunk_overlap,
            "top_k_results": settings.rag.top_k_results
        }
    }


def print_config_summary() -> None:
    """Печать сводки конфигурации"""
    summary = get_config_summary()
    print("\n" + "=" * 50)
    print("RAG Service Configuration Summary")
    print("=" * 50)

    for category, config in summary.items():
        if isinstance(config, dict):
            print(f"\n{category.upper()}:")
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"{category}: {config}")

    print("=" * 50 + "\n")


# Пример использования настроек в других модулях
if __name__ == "__main__":
    # Печать сводки конфигурации
    print_config_summary()

    # Проверка валидации настроек
    errors = settings.validate_settings()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")

    # Пример получения URL сервисов
    print("\nService URLs:")
    for service, url in settings.get_service_urls().items():
        print(f"  {service}: {url}")