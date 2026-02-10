# src/services/minio_service.py
import io
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, BinaryIO, Tuple
from minio import Minio
from minio.error import S3Error
from minio.commonconfig import Tags
from minio.deleteobjects import DeleteObject
import mimetypes
from urllib.parse import urlparse

from app.config.settings import settings


class MinioService:
    """Сервис для работы с MinIO (объектное хранилище)"""

    def __init__(self):
        """Инициализация клиента MinIO"""
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )

        # Основной bucket для RAG документов
        self.default_bucket = settings.MINIO_BUCKET

        # Создаем необходимые bucket'ы при инициализации
        self._ensure_buckets()

    def _ensure_buckets(self) -> None:
        """Создание необходимых bucket'ов если они не существуют"""
        required_buckets = [
            self.default_bucket,  # Основной bucket для документов
            "pdf-originals",  # Оригинальные PDF файлы
            "pdf-analysis",  # JSON результаты анализа PDF
            "extracted-images",  # Извлеченные изображения
            "pdf-thumbnails",  # Миниатюры PDF
            "processed-docs",  # Обработанные документы
            "mineru-images",  # Изображения из MinerU
            "rag-embeddings",  # Векторные эмбеддинги
            "rag-metadata",  # Метаданные документов
            "temp-uploads",  # Временные загрузки
        ]

        for bucket_name in required_buckets:
            try:
                if not self.client.bucket_exists(bucket_name):
                    self.client.make_bucket(bucket_name)
                    print(f"Bucket создан: {bucket_name}")
            except S3Error as e:
                print(f"Ошибка при создании bucket {bucket_name}: {e}")

    def calculate_file_hash(self, file_data: Union[bytes, BinaryIO]) -> str:
        """Вычисление MD5 хэша файла"""
        if isinstance(file_data, bytes):
            return hashlib.md5(file_data).hexdigest()
        else:
            # Если это file-like объект, читаем его
            file_data.seek(0)
            md5_hash = hashlib.md5()
            for chunk in iter(lambda: file_data.read(4096), b""):
                md5_hash.update(chunk)
            file_data.seek(0)
            return md5_hash.hexdigest()

    def upload_file(
            self,
            bucket_name: str,
            file_name: str,
            file_data: Union[bytes, BinaryIO],
            content_type: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Загрузка файла в MinIO

        Args:
            bucket_name: Название bucket
            file_name: Имя файла
            file_data: Данные файла (bytes или file-like объект)
            content_type: MIME-тип файла
            metadata: Дополнительные метаданные
            tags: Теги для файла

        Returns:
            Dict с информацией о загруженном файле
        """
        try:
            # Определяем content-type если не указан
            if content_type is None:
                content_type, _ = mimetypes.guess_type(file_name)
                if content_type is None:
                    content_type = "application/octet-stream"

            # Преобразуем bytes в BytesIO если нужно
            if isinstance(file_data, bytes):
                data_stream = io.BytesIO(file_data)
                file_size = len(file_data)
            else:
                data_stream = file_data
                data_stream.seek(0, 2)  # Перемещаемся в конец
                file_size = data_stream.tell()
                data_stream.seek(0)  # Возвращаемся в начало

            # Вычисляем хэш файла
            file_hash = self.calculate_file_hash(data_stream)

            # Добавляем timestamp к имени файла для уникальности
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_name = f"{timestamp}_{file_hash[:8]}_{file_name}"

            # Подготавливаем метаданные
            minio_metadata = {
                "original-filename": file_name,
                "upload-timestamp": timestamp,
                "file-hash": file_hash,
                "content-type": content_type
            }

            if metadata:
                minio_metadata.update(metadata)

            # Загружаем файл
            result = self.client.put_object(
                bucket_name=bucket_name,
                object_name=unique_name,
                data=data_stream,
                length=file_size,
                content_type=content_type,
                metadata=minio_metadata
            )

            # Добавляем теги если есть
            if tags:
                self.client.set_object_tags(
                    bucket_name=bucket_name,
                    object_name=unique_name,
                    tags=Tags.new_object_tags(tags)
                )

            # Формируем информацию о загруженном файле
            file_info = {
                "bucket": bucket_name,
                "object_name": unique_name,
                "original_filename": file_name,
                "file_hash": file_hash,
                "file_size": file_size,
                "content_type": content_type,
                "etag": result.etag,
                "version_id": result.version_id,
                "minio_path": f"{bucket_name}/{unique_name}",
                "upload_time": timestamp
            }

            return file_info

        except S3Error as e:
            raise Exception(f"Ошибка загрузки файла в MinIO: {e}")
        except Exception as e:
            raise Exception(f"Неизвестная ошибка при загрузке файла: {e}")

    def download_file(self, bucket_name: str, object_name: str) -> bytes:
        """
        Скачивание файла из MinIO

        Args:
            bucket_name: Название bucket
            object_name: Имя объекта в MinIO

        Returns:
            bytes содержимого файла
        """
        try:
            response = self.client.get_object(bucket_name, object_name)
            file_data = response.read()
            response.close()
            response.release_conn()
            return file_data
        except S3Error as e:
            raise Exception(f"Ошибка скачивания файла из MinIO: {e}")

    def download_file_stream(self, bucket_name: str, object_name: str) -> BinaryIO:
        """
        Скачивание файла из MinIO в виде stream

        Args:
            bucket_name: Название bucket
            object_name: Имя объекта в MinIO

        Returns:
            BytesIO stream с содержимым файла
        """
        try:
            file_data = self.download_file(bucket_name, object_name)
            return io.BytesIO(file_data)
        except Exception as e:
            raise Exception(f"Ошибка при создании stream: {e}")

    def file_exists(self, bucket_name: str, object_name: str) -> bool:
        """Проверка существования файла в MinIO"""
        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            else:
                raise Exception(f"Ошибка при проверке существования файла: {e}")

    def get_file_info(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """Получение информации о файле"""
        try:
            stat = self.client.stat_object(bucket_name, object_name)

            # Получаем метаданные
            metadata = {}
            if hasattr(stat, 'metadata'):
                metadata = {k.lower(): v for k, v in stat.metadata.items()}

            # Получаем теги
            try:
                tags_response = self.client.get_object_tags(bucket_name, object_name)
                tags = tags_response.tags if tags_response else {}
            except:
                tags = {}

            file_info = {
                "bucket": bucket_name,
                "object_name": object_name,
                "size": stat.size,
                "etag": stat.etag,
                "last_modified": stat.last_modified,
                "content_type": stat.content_type,
                "metadata": metadata,
                "tags": tags,
                "version_id": stat.version_id
            }

            return file_info

        except S3Error as e:
            raise Exception(f"Ошибка при получении информации о файле: {e}")

    def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """Удаление файла из MinIO"""
        try:
            self.client.remove_object(bucket_name, object_name)
            return True
        except S3Error as e:
            raise Exception(f"Ошибка при удалении файла: {e}")

    def delete_files(self, bucket_name: str, object_names: List[str]) -> List[Dict[str, Any]]:
        """Массовое удаление файлов из MinIO"""
        try:
            # Создаем список объектов для удаления
            delete_objects = [DeleteObject(obj_name) for obj_name in object_names]

            # Выполняем удаление
            errors = self.client.remove_objects(bucket_name, delete_objects)

            results = []
            for error in errors:
                results.append({
                    "object_name": error.object_name,
                    "error_code": error.code,
                    "error_message": error.message,
                    "success": False
                })

            # Для успешно удаленных объектов
            success_count = len(object_names) - len(results)
            for obj_name in object_names:
                if not any(r["object_name"] == obj_name for r in results):
                    results.append({
                        "object_name": obj_name,
                        "success": True
                    })

            return results

        except S3Error as e:
            raise Exception(f"Ошибка при массовом удалении файлов: {e}")

    def list_files(
            self,
            bucket_name: str,
            prefix: Optional[str] = None,
            recursive: bool = True,
            limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Получение списка файлов в bucket

        Args:
            bucket_name: Название bucket
            prefix: Префикс для фильтрации
            recursive: Рекурсивный поиск
            limit: Максимальное количество результатов

        Returns:
            Список информации о файлах
        """
        try:
            objects = self.client.list_objects(
                bucket_name=bucket_name,
                prefix=prefix,
                recursive=recursive
            )

            files = []
            for obj in objects:
                if len(files) >= limit:
                    break

                file_info = {
                    "bucket": bucket_name,
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag,
                    "is_dir": obj.is_dir
                }
                files.append(file_info)

            return files

        except S3Error as e:
            raise Exception(f"Ошибка при получении списка файлов: {e}")

    def search_files_by_hash(self, bucket_name: str, file_hash: str) -> List[Dict[str, Any]]:
        """Поиск файлов по хэшу"""
        try:
            # Ищем файлы с хэшем в метаданных
            all_files = self.list_files(bucket_name, recursive=True)

            matching_files = []
            for file_info in all_files:
                try:
                    file_metadata = self.get_file_info(bucket_name, file_info["object_name"])
                    if file_metadata.get("metadata", {}).get("file-hash") == file_hash:
                        matching_files.append(file_metadata)
                except:
                    continue

            return matching_files

        except Exception as e:
            raise Exception(f"Ошибка при поиске файлов по хэшу: {e}")

    def get_presigned_url(
            self,
            bucket_name: str,
            object_name: str,
            expires: timedelta = timedelta(hours=1),
            response_headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Генерация предварительно подписанного URL для доступа к файлу

        Args:
            bucket_name: Название bucket
            object_name: Имя объекта
            expires: Время жизни URL
            response_headers: Дополнительные HTTP заголовки

        Returns:
            Предварительно подписанный URL
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_name,
                expires=int(expires.total_seconds()),
                response_headers=response_headers
            )
            return url
        except S3Error as e:
            raise Exception(f"Ошибка при генерации подписанного URL: {e}")

    def upload_json(
            self,
            bucket_name: str,
            object_name: str,
            data: Union[Dict, List],
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Загрузка JSON данных в MinIO

        Args:
            bucket_name: Название bucket
            object_name: Имя объекта
            data: Данные для сохранения (dict или list)
            metadata: Дополнительные метаданные

        Returns:
            Информация о загруженном файле
        """
        try:
            # Конвертируем данные в JSON строку
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            json_bytes = json_str.encode('utf-8')

            # Добавляем информацию о JSON в метаданные
            json_metadata = {
                "data-type": "json",
                "json-schema": "generic"
            }
            if metadata:
                json_metadata.update(metadata)

            # Загружаем JSON
            return self.upload_file(
                bucket_name=bucket_name,
                file_name=object_name,
                file_data=json_bytes,
                content_type="application/json",
                metadata=json_metadata
            )

        except Exception as e:
            raise Exception(f"Ошибка при загрузке JSON: {e}")

    def download_json(self, bucket_name: str, object_name: str) -> Union[Dict, List]:
        """
        Скачивание JSON данных из MinIO

        Args:
            bucket_name: Название bucket
            object_name: Имя объекта

        Returns:
            Распарсенные JSON данные
        """
        try:
            json_bytes = self.download_file(bucket_name, object_name)
            json_str = json_bytes.decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            raise Exception(f"Ошибка при скачивании JSON: {e}")

    def upload_image(
            self,
            bucket_name: str,
            image_name: str,
            image_data: Union[bytes, BinaryIO],
            image_format: str = "jpeg",
            quality: int = 85,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Загрузка изображения в MinIO с оптимизацией

        Args:
            bucket_name: Название bucket
            image_name: Имя изображения
            image_data: Данные изображения
            image_format: Формат изображения (jpeg, png, etc.)
            quality: Качество сжатия (для JPEG)
            metadata: Дополнительные метаданные

        Returns:
            Информация о загруженном изображении
        """
        try:
            # Определяем content-type
            content_type = f"image/{image_format.lower()}"

            # Добавляем информацию об изображении в метаданные
            image_metadata = {
                "data-type": "image",
                "image-format": image_format,
                "quality": str(quality)
            }
            if metadata:
                image_metadata.update(metadata)

            # Загружаем изображение
            return self.upload_file(
                bucket_name=bucket_name,
                file_name=image_name,
                file_data=image_data,
                content_type=content_type,
                metadata=image_metadata
            )

        except Exception as e:
            raise Exception(f"Ошибка при загрузке изображения: {e}")

    def upload_pdf(
            self,
            bucket_name: str,
            pdf_name: str,
            pdf_data: Union[bytes, BinaryIO],
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Загрузка PDF файла в MinIO

        Args:
            bucket_name: Название bucket
            pdf_name: Имя PDF файла
            pdf_data: Данные PDF
            metadata: Дополнительные метаданные

        Returns:
            Информация о загруженном PDF
        """
        try:
            # Добавляем информацию о PDF в метаданные
            pdf_metadata = {
                "data-type": "pdf",
                "document-type": "pdf"
            }
            if metadata:
                pdf_metadata.update(metadata)

            # Загружаем PDF
            return self.upload_file(
                bucket_name=bucket_name,
                file_name=pdf_name,
                file_data=pdf_data,
                content_type="application/pdf",
                metadata=pdf_metadata
            )

        except Exception as e:
            raise Exception(f"Ошибка при загрузке PDF: {e}")

    def copy_file(
            self,
            source_bucket: str,
            source_object: str,
            dest_bucket: str,
            dest_object: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Копирование файла между bucket'ами или объектами

        Args:
            source_bucket: Исходный bucket
            source_object: Исходный объект
            dest_bucket: Целевой bucket
            dest_object: Целевой объект
            metadata: Новые метаданные

        Returns:
            Информация о скопированном файле
        """
        try:
            # Копируем объект
            copy_result = self.client.copy_object(
                bucket_name=dest_bucket,
                object_name=dest_object,
                source=f"/{source_bucket}/{source_object}",
                metadata=metadata,
                metadata_directive="REPLACE" if metadata else "COPY"
            )

            # Получаем информацию о скопированном файле
            return self.get_file_info(dest_bucket, dest_object)

        except S3Error as e:
            raise Exception(f"Ошибка при копировании файла: {e}")

    def get_bucket_size(self, bucket_name: str) -> Dict[str, Any]:
        """Получение информации о размере bucket"""
        try:
            files = self.list_files(bucket_name, recursive=True)

            total_size = 0
            file_count = 0

            for file_info in files:
                if not file_info.get("is_dir", True):
                    total_size += file_info.get("size", 0)
                    file_count += 1

            return {
                "bucket": bucket_name,
                "file_count": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_size_gb": round(total_size / (1024 * 1024 * 1024), 4)
            }

        except Exception as e:
            raise Exception(f"Ошибка при получении размера bucket: {e}")

    def get_all_buckets_info(self) -> List[Dict[str, Any]]:
        """Получение информации о всех bucket'ах"""
        try:
            buckets = self.client.list_buckets()
            buckets_info = []

            for bucket in buckets:
                try:
                    bucket_info = self.get_bucket_size(bucket.name)
                    buckets_info.append(bucket_info)
                except:
                    # Если не удалось получить размер, добавляем базовую информацию
                    buckets_info.append({
                        "bucket": bucket.name,
                        "file_count": 0,
                        "total_size_bytes": 0,
                        "creation_date": bucket.creation_date
                    })

            return buckets_info

        except Exception as e:
            raise Exception(f"Ошибка при получении информации о bucket'ах: {e}")

    def create_folder(self, bucket_name: str, folder_path: str) -> bool:
        """Создание папки в bucket (пустого объекта с завершающим слэшем)"""
        try:
            # Убедимся, что путь заканчивается на /
            if not folder_path.endswith('/'):
                folder_path += '/'

            # Создаем пустой объект для представления папки
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=folder_path,
                data=io.BytesIO(b''),
                length=0,
                content_type="application/x-directory"
            )

            return True
        except S3Error as e:
            raise Exception(f"Ошибка при создании папки: {e}")

    def list_folders(self, bucket_name: str, prefix: str = "") -> List[str]:
        """Получение списка папок в bucket"""
        try:
            files = self.list_files(bucket_name, prefix=prefix, recursive=False)
            folders = []

            for file_info in files:
                if file_info.get("is_dir"):
                    folders.append(file_info["object_name"])

            return folders

        except Exception as e:
            raise Exception(f"Ошибка при получении списка папок: {e}")

    def cleanup_old_files(
            self,
            bucket_name: str,
            days_old: int = 30,
            prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Удаление старых файлов из bucket

        Args:
            bucket_name: Название bucket
            days_old: Удалять файлы старше этого количества дней
            prefix: Префикс для фильтрации

        Returns:
            Статистика удаления
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            files = self.list_files(bucket_name, prefix=prefix, recursive=True)

            to_delete = []
            deleted_count = 0
            error_count = 0

            for file_info in files:
                if not file_info.get("is_dir"):
                    last_modified = file_info.get("last_modified")
                    if last_modified and last_modified < cutoff_date:
                        to_delete.append(file_info["object_name"])

            # Удаляем файлы партиями
            batch_size = 100
            for i in range(0, len(to_delete), batch_size):
                batch = to_delete[i:i + batch_size]
                try:
                    results = self.delete_files(bucket_name, batch)

                    for result in results:
                        if result.get("success"):
                            deleted_count += 1
                        else:
                            error_count += 1
                except Exception as e:
                    print(f"Ошибка при удалении batch {i}: {e}")
                    error_count += len(batch)

            return {
                "bucket": bucket_name,
                "total_scanned": len(files),
                "marked_for_deletion": len(to_delete),
                "deleted": deleted_count,
                "errors": error_count,
                "cutoff_date": cutoff_date.isoformat()
            }

        except Exception as e:
            raise Exception(f"Ошибка при очистке старых файлов: {e}")

    def get_file_statistics(self, bucket_name: str) -> Dict[str, Any]:
        """Получение статистики по файлам в bucket"""
        try:
            files = self.list_files(bucket_name, recursive=True)

            file_types = {}
            total_size = 0
            file_count = 0

            for file_info in files:
                if not file_info.get("is_dir"):
                    file_count += 1
                    size = file_info.get("size", 0)
                    total_size += size

                    # Определяем тип файла по расширению
                    object_name = file_info["object_name"]
                    extension = os.path.splitext(object_name)[1].lower()

                    if extension:
                        if extension in file_types:
                            file_types[extension]["count"] += 1
                            file_types[extension]["total_size"] += size
                        else:
                            file_types[extension] = {
                                "count": 1,
                                "total_size": size
                            }

            return {
                "bucket": bucket_name,
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "avg_file_size": round(total_size / file_count, 2) if file_count > 0 else 0
            }

        except Exception as e:
            raise Exception(f"Ошибка при получении статистики файлов: {e}")


# Создаем экземпляр сервиса для использования в других модулях
minio_service = MinioService()


# Утилитарные функции для работы с MinIO путями
def parse_minio_path(minio_path: str) -> Tuple[str, str]:
    """Парсинг MinIO пути в bucket и object_name"""
    if '/' not in minio_path:
        raise ValueError(f"Invalid MinIO path format: {minio_path}")

    bucket, object_name = minio_path.split('/', 1)
    return bucket, object_name


def generate_minio_path(bucket: str, object_name: str) -> str:
    """Генерация MinIO пути из bucket и object_name"""
    return f"{bucket}/{object_name}"


# Декоратор для обработки ошибок MinIO
def handle_minio_errors(func):
    """Декоратор для обработки ошибок MinIO"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except S3Error as e:
            error_info = {
                "error_type": "MinIO Error",
                "error_code": e.code if hasattr(e, 'code') else "UNKNOWN",
                "error_message": str(e),
                "function": func.__name__
            }
            raise Exception(json.dumps(error_info))
        except Exception as e:
            error_info = {
                "error_type": "General Error",
                "error_message": str(e),
                "function": func.__name__
            }
            raise Exception(json.dumps(error_info))

    return wrapper


# Класс для работы с временными файлами в MinIO
class MinioTempFileManager:
    """Менеджер временных файлов в MinIO"""

    def __init__(self, minio_service: MinioService, temp_bucket: str = "temp-uploads"):
        self.minio_service = minio_service
        self.temp_bucket = temp_bucket

        # Создаем временный bucket если его нет
        if not self.minio_service.client.bucket_exists(temp_bucket):
            self.minio_service.client.make_bucket(temp_bucket)

    def create_temp_file(self, file_data: bytes, prefix: str = "temp") -> str:
        """Создание временного файла в MinIO"""
        temp_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(file_data).hexdigest()[:8]}"
        temp_name = f"{temp_id}.tmp"

        self.minio_service.upload_file(
            bucket_name=self.temp_bucket,
            file_name=temp_name,
            file_data=file_data
        )

        return f"{self.temp_bucket}/{temp_name}"

    def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """Очистка старых временных файлов"""
        stats = self.minio_service.cleanup_old_files(
            bucket_name=self.temp_bucket,
            days_old=older_than_hours / 24
        )
        return stats.get("deleted", 0)

    def get_temp_file(self, minio_path: str) -> bytes:
        """Получение временного файла"""
        bucket, object_name = parse_minio_path(minio_path)
        return self.minio_service.download_file(bucket, object_name)

    def delete_temp_file(self, minio_path: str) -> bool:
        """Удаление временного файла"""
        bucket, object_name = parse_minio_path(minio_path)
        return self.minio_service.delete_file(bucket, object_name)


# Пример использования сервиса
if __name__ == "__main__":
    # Создание экземпляра сервиса
    service = MinioService()

    # Пример загрузки файла
    test_data = b"Hello, MinIO!"
    result = service.upload_file(
        bucket_name="test-bucket",
        file_name="test.txt",
        file_data=test_data,
        content_type="text/plain",
        metadata={"author": "test-user", "description": "Test file"}
    )

    print(f"Файл загружен: {result['minio_path']}")

    # Пример скачивания файла
    downloaded = service.download_file("test-bucket", result["object_name"])
    print(f"Скачано: {downloaded.decode('utf-8')}")

    # Пример получения информации о bucket'ах
    buckets_info = service.get_all_buckets_info()
    print(f"Bucket'ы: {json.dumps(buckets_info, indent=2)}")