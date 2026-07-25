# Data Model: Application Settings

## Purpose
Модели конфигурации приложения на базе Pydantic `BaseSettings`. Определяют все параметры подключения к внешним сервисам и внутренние настройки приложения. Загружаются из переменных окружения и `.env`-файла. Агрегируются в единый класс `Settings` для централизованного доступа.

Расположение: `app/config/settings.py`.

## Schema

```python
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class S3Settings(BaseSettings):
    """S3/MinIO storage configuration."""
    S3_URL: str = "http://localhost:9000"          # S3 endpoint URL
    S3_ACCESS_KEY: str = "minio"                    # S3 access key
    S3_SECRET_KEY: str = "minio123"                 # S3 secret key
    S3_VERIFY_TLS: bool = False                     # Verify TLS certificates
    S3_BUCKET_NAME: str = "pdf-processing"          # Default bucket name
    MINIO_ROOT_USER: str = "minioadmin"              # MinIO root user
    MINIO_ROOT_PASSWORD: str = "minioadmin"          # MinIO root password
    MINIO_ENDPOINT: str = "minio:9000"               # MinIO internal Docker endpoint
    MINIO_BUCKET: str = "pdf-processing"             # MinIO bucket name

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""
    QDRANT_HOST: str = "localhost"                  # Qdrant host
    QDRANT_PORT: int = 6333                         # Qdrant HTTP port
    QDRANT_GRPC_PORT: int = 6334                    # Qdrant gRPC port
    QDRANT_API_KEY: Optional[str] = None             # Qdrant API key
    QDRANT_COLLECTION_NAME: str = "documents"        # Default collection name
    host: str = "0.0.0.0"                           # Qdrant service host (дублирующее поле)
    port: int = 8000                                # Qdrant service port (дублирующее поле)
    api_key: str = ""                               # Qdrant service API key (дублирующее поле)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class MinerUSettings(BaseSettings):
    """MinerU document processing service configuration."""
    MINERU_HOST: str = "http://localhost"            # MinerU service host
    MINERU_PORT: int = 8001                          # MinerU service port
    MINERU_TIMEOUT: int = 300                        # Request timeout (seconds)
    MINERU_MAX_FILE_SIZE: int = 52428800             # Max upload file size (50MB)
    MINERU_CACHE_TTL: int = 3600                     # Cache TTL (seconds)
    MODELSCOPE_CACHE: str = "/app/models"            # ModelScope cache directory
    MINERU_BACKEND: str = "pipeline"                 # Backend: "pipeline" or "vlm"
    MINERU_METHOD: str = "auto"                      # Method: "auto", "txt", "ocr"
    MINERU_LANG: str = "ru"                          # Default document language
    MINERU_FORMULA_ENABLE: bool = True                # Enable formula processing
    MINERU_TABLE_ENABLE: bool = True                  # Enable table processing

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class EmbeddingSettings(BaseSettings):
    """Embedding service configuration."""
    EMBEDDING_BASE_URL: str = "http://192.168.19.127:10115/embedding"  # Embedding service URL
    EMBEDDING_TIMEOUT: int = 30                                         # Request timeout (seconds)
    EMBEDDING_MODEL: str = "qwen3-emb"                                  # Embedding model name

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class RerankerSettings(BaseSettings):
    """Reranker service configuration."""
    RERANKER_BASE_URL: str = "http://192.168.19.127:10115/reranker"    # Reranker service URL
    RERANKER_TIMEOUT: int = 30                                         # Request timeout (seconds)
    RERANKER_TOP_N: int = 100                                          # Default number of top results

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class LLMSettings(BaseSettings):
    """LLM service configuration."""
    LLM_BASE_URL: str = "http://192.168.19.127:8888/v1"                # LLM API URL (OpenAI-compatible)
    LLM_API_KEY: str = "EMPTY"                                          # LLM API key
    LLM_MODEL_NAME: str = "Qwen/Qwen3-VL-32B-Thinking"                 # Model name
    LLM_MAX_TOKENS: int = 2048                                          # Max response tokens
    LLM_TEMPERATURE: float = 0.7                                        # Generation temperature

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class AppSettings(BaseSettings):
    """Main application configuration."""
    HOST: str = "0.0.0.0"                            # Application host
    PORT: int = 8000                                  # Application port
    DEBUG: bool = False                               # Debug mode
    LOG_LEVEL: str = "INFO"                           # Logging level
    MAX_FILE_SIZE: int = 52428800                     # Max upload file size (50MB)
    CACHE_TTL: int = 3600                             # Cache TTL (seconds)
    TEMP_DIR: str = "/tmp/pdf_processing"             # Temporary directory
    CORS_ORIGINS: List[str] = ["*"]                   # Allowed CORS origins

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
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    app: AppSettings = Field(default_factory=AppSettings)

    # Direct access aliases for backward compatibility
    S3_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minio"
    S3_SECRET_KEY: str = "minio123"
    S3_VERIFY_TLS: bool = False
    S3_BUCKET_NAME: str = "pdf-processing"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "documents"
    MINERU_HOST: str = "http://localhost"
    MINERU_PORT: int = 8001
    MINERU_TIMEOUT: int = 300
    MODELSCOPE_CACHE: str = "/app/models"
    EMBEDDING_BASE_URL: str = "http://192.168.19.127:10114/embedding"
    EMBEDDING_TIMEOUT: int = 30
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE: int = 52428800
    CACHE_TTL: int = 3600

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
        """Get S3 endpoint without http(s):// prefix for Minio client."""
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
```

## Storage
- **Источник**: переменные окружения (`os.environ`) и файл `.env` (автоматическая загрузка через `SettingsConfigDict(env_file=".env")`)
- **В памяти**: глобальный экземпляр `settings = Settings()` в `config/settings.py` (singleton)
- **Экспорт на уровне модуля**: выборочные поля `Settings` реэкспортируются как переменные модуля для обратной совместимости:
  ```python
  S3_URL = settings.S3_URL
  S3_ACCESS_KEY = settings.S3_ACCESS_KEY
  # ... и так далее
  ```

## Relationships
- **`Settings` (агрегат)** → используется во всех модулях приложения: `api.py`, `qdrant_client_api.py`, `minio_client.py`, `qwen3_emb_client.py`, `reranker_client.py`
- **`S3Settings`** → `MinioClient.__init__()` — endpoint, access_key, secret_key, secure, bucket_name
- **`QdrantSettings`** → `get_qdrant_client()` — host, port, api_key, collection_name
- **`MinerUSettings`** → `MinerUClient.__init__()` — host, port, timeout, backend, method, lang
- **`EmbeddingSettings`** → `EmbeddingClient.__init__()` — base_url, timeout
- **`RerankerSettings`** → `RerankerClient.__init__()` — base_url, timeout
- **`LLMSettings`** → `LLMClient.__init__()` — base_url, api_key, model_name, max_tokens, temperature
- **`AppSettings`** → `uvicorn.run()` — host, port; CORS middleware setup; temp_dir для временных файлов
- **`Settings.mineru_url`**, **`Settings.qdrant_url`**, **`Settings.s3_endpoint_clean`**, **`Settings.s3_secure`** — computed properties для получения производных значений
- **Обратная совместимость**: модульные переменные (`S3_URL`, `QDRANT_HOST`, ...) позволяют старому коду импортировать напрямую: `from app.config.settings import S3_URL`

## Constraints
- **`SettingsConfigDict(env_file=".env")`** — все Settings-классы загружают `.env` независимо; путь разрешается относительно рабочей директории (CWD)
- **`SettingsConfigDict(extra="ignore")`** — все классы игнорируют неизвестные переменные окружения (не вызывают ошибок)
- **Дублирование полей**: `Settings` (агрегат) содержит как вложенные объекты (`settings.s3.S3_URL`), так и прямые алиасы (`settings.S3_URL`). Дефолтные значения дублируются, но **значения из переменных окружения для дублирующих полей НЕ СВЯЗАНЫ** — `settings.S3_URL` и `settings.s3.S3_URL` загружаются независимо и могут расходиться
- **`EMBEDDING_BASE_URL`**: значение по умолчанию в `EmbeddingSettings` — `http://192.168.19.127:10115/embedding`, но в `Settings` (агрегат) — `http://192.168.19.127:10114/embedding` (разные порты! Расхождение между вложенным и агрегатным дефолтами.)
- **`QdrantSettings`**: содержит дублирующие поля `host`/`port`/`api_key` (camelCase) помимо `QDRANT_HOST`/`QDRANT_PORT`/`QDRANT_API_KEY` (UPPER_CASE). Разные имена для одних и тех же настроек.
- **`CORS_ORIGINS`**: default `["*"]` — разрешает все origins, потенциальная проблема безопасности в production
- **`LLM_API_KEY`**: default `"EMPTY"` — заглушка, в production должна быть переменная окружения
- **`MINERU_LANG`**: default `"ru"` — жёсткая привязка к русскому языку

### Граничные случаи
- **Отсутствующий `.env` файл**: используются defaults (`SettingsConfigDict(env_file=".env")` не вызывает ошибки при отсутствии файла — pydantic-settings silently skips)
- **Конфликт env и default**: переменная окружения всегда переопределяет default
- **Несовпадение портов в `EmbeddingSettings`**: при обращении через агрегат (`settings.embedding.EMBEDDING_BASE_URL`) порт 10115, через алиас (`settings.EMBEDDING_BASE_URL`) порт 10114
- **`extra="ignore"`**: лишние переменные в `.env` игнорируются без ошибок (silently)
- **Пустые строки для числовых полей**: pydantic-settings парсит `PORT=""` как ошибку валидации (int expected)

## Examples

### Использование агрегированного settings
```python
from app.config.settings import settings

# Доступ через вложенные объекты
base_url = settings.embedding.EMBEDDING_BASE_URL  # "http://192.168.19.127:10115/embedding"
timeout = settings.embedding.EMBEDDING_TIMEOUT    # 30

# Доступ через computed property
mineru_url = settings.mineru_url  # "http://localhost:8001"

# Доступ через прямой алиас (обратная совместимость)
host = settings.HOST  # "0.0.0.0"
```

### Переопределение через переменные окружения
```bash
# .env
EMBEDDING_BASE_URL=http://prod-embedding:10115/embedding
LLM_API_KEY=sk-real-api-key
LLM_TEMPERATURE=0.3
```
```python
# settings.embedding.EMBEDDING_BASE_URL → "http://prod-embedding:10115/embedding"
# settings.llm.LLM_API_KEY → "sk-real-api-key"
# settings.llm.LLM_TEMPERATURE → 0.3  # int из env → float через pydantic coercion
```

### Потенциальная проблема расхождения
```python
# Если установлена переменная окружения:
#   EMBEDDING_BASE_URL=http://custom:9999/embedding
# Тогда:
settings.embedding.EMBEDDING_BASE_URL  # → "http://custom:9999/embedding"  (OK, из env)
settings.EMBEDDING_BASE_URL            # → "http://custom:9999/embedding"  (OK, из env)
# Но если переменная НЕ установлена:
settings.embedding.EMBEDDING_BASE_URL  # → "http://192.168.19.127:10115/embedding"
settings.EMBEDDING_BASE_URL            # → "http://192.168.19.127:10114/embedding"  # РАСХОЖДЕНИЕ!
```

## Exceptions
- **Отклонение от Constitution N2 (структура модулей)**: Settings расположены в `config/settings.py`, что соответствует структуре N2.
- **Дублирование полей в агрегате `Settings`**: Наличие и вложенных объектов, и прямых алиасов с одинаковыми именами — архитектурный дефект, создающий риск расхождения значений. Прямые алиасы загружаются из env независимо от вложенных объектов и имеют собственные дефолты.
- **Разные дефолтные порты**: `EmbeddingSettings.EMBEDDING_BASE_URL` (порт 10115) и `Settings.EMBEDDING_BASE_URL` (порт 10114) имеют разные значения по умолчанию.
- **Отсутствие `token` в `model_config`**: pydantic-settings не имеет явного `case_sensitive` или `env_prefix` — используется поведение по умолчанию (case-insensitive matching).
