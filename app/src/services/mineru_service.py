# src/services/mineru_service.py
import requests
import json
import base64
import io
import os
import tempfile
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config.settings import settings

logger = logging.getLogger(__name__)


class MinerUService:
    """Сервис для взаимодействия с MinerU API (анализ документов)"""

    def __init__(self):
        """Инициализация сервиса MinerU"""
        self.base_url = settings.mineru.url
        self.timeout = settings.mineru.timeout
        self.max_file_size = settings.mineru.max_file_size

        # Кэш для уже обработанных файлов (file_hash -> result)
        self._cache = {}
        self._cache_ttl = 3600  # 1 час

        # Проверяем соединение при инициализации
        self._check_connection()

    def _check_connection(self) -> bool:
        """Проверка соединения с сервисом MinerU"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"MinerU сервис доступен: {response.json()}")
                return True
            else:
                logger.warning(f"MinerU сервис недоступен: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ошибка подключения к MinerU: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья MinerU сервиса"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "details": response.json(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def calculate_file_hash(self, file_bytes: bytes) -> str:
        """Вычисление MD5 хэша файла"""
        return hashlib.md5(file_bytes).hexdigest()

    def _check_cache(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Проверка кэша по хэшу файла"""
        if file_hash in self._cache:
            cached_item = self._cache[file_hash]
            if time.time() - cached_item["timestamp"] < self._cache_ttl:
                logger.info(f"Найден результат в кэше для хэша {file_hash}")
                return cached_item["result"]
            else:
                # Удаляем устаревший кэш
                del self._cache[file_hash]
        return None

    def _save_to_cache(self, file_hash: str, result: Dict[str, Any]):
        """Сохранение результата в кэш"""
        self._cache[file_hash] = {
            "result": result,
            "timestamp": time.time()
        }

    def _prepare_file_for_upload(
            self,
            file_data: Union[bytes, str, Path],
            filename: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """
        Подготовка файла для загрузки в MinerU

        Args:
            file_data: Данные файла (bytes, путь или base64 строка)
            filename: Имя файла (если не указано, генерируется)

        Returns:
            Tuple[bytes, str]: (данные файла, имя файла)
        """
        try:
            # Если это путь
            if isinstance(file_data, (str, Path)):
                path = Path(file_data)
                if not path.exists():
                    raise FileNotFoundError(f"Файл не найден: {file_data}")

                with open(path, 'rb') as f:
                    file_bytes = f.read()

                if filename is None:
                    filename = path.name

            # Если это base64 строка
            elif isinstance(file_data, str) and file_data.startswith('data:'):
                # Извлекаем base64 часть
                if ';base64,' in file_data:
                    header, data = file_data.split(';base64,', 1)
                    file_bytes = base64.b64decode(data)

                    if filename is None:
                        # Пытаемся определить расширение из заголовка
                        content_type = header.replace('data:', '')
                        extension = self._get_extension_from_mime(content_type)
                        filename = f"file{extension}"
                else:
                    raise ValueError("Некорректный формат base64 строки")

            # Если это bytes
            elif isinstance(file_data, bytes):
                file_bytes = file_data
                if filename is None:
                    filename = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            else:
                raise TypeError(f"Неподдерживаемый тип данных: {type(file_data)}")

            # Проверяем размер файла
            if len(file_bytes) > self.max_file_size:
                raise ValueError(
                    f"Размер файла ({len(file_bytes)} bytes) превышает максимальный "
                    f"допустимый размер ({self.max_file_size} bytes)"
                )

            return file_bytes, filename

        except Exception as e:
            logger.error(f"Ошибка подготовки файла: {e}")
            raise

    def _get_extension_from_mime(self, mime_type: str) -> str:
        """Получение расширения файла из MIME типа"""
        mime_to_ext = {
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'application/pdf': '.pdf',
            'text/plain': '.txt',
        }
        return mime_to_ext.get(mime_type.lower(), '.bin')

    def analyze_document(
            self,
            file_data: Union[bytes, str, Path],
            filename: Optional[str] = None,
            use_cache: bool = True,
            analyze_type: str = "full"  # full, tables, structured
    ) -> Dict[str, Any]:
        """
        Анализ документа через MinerU API

        Args:
            file_data: Данные файла (bytes, путь или base64 строка)
            filename: Имя файла
            use_cache: Использовать кэш
            analyze_type: Тип анализа (full, tables, structured)

        Returns:
            Результат анализа от MinerU
        """
        start_time = time.time()

        try:
            # Подготавливаем файл
            file_bytes, processed_filename = self._prepare_file_for_upload(file_data, filename)

            # Вычисляем хэш
            file_hash = self.calculate_file_hash(file_bytes)

            # Проверяем кэш
            if use_cache:
                cached_result = self._check_cache(file_hash)
                if cached_result:
                    cached_result["_cached"] = True
                    cached_result["_processing_time"] = time.time() - start_time
                    return cached_result

            # Определяем конечный URL в зависимости от типа анализа
            if analyze_type == "tables":
                endpoint = "/analyze/tables-with-images"
            elif analyze_type == "structured":
                endpoint = "/analyze/structured-with-images"
            else:
                endpoint = "/analyze/with-images"

            url = f"{self.base_url}{endpoint}"

            # Создаем временный файл для отправки
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(processed_filename)[1]) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                # Отправляем запрос в MinerU
                logger.info(f"Отправка файла {processed_filename} в MinerU (тип анализа: {analyze_type})")

                with open(tmp_path, 'rb') as f:
                    files = {'file': (processed_filename, f)}
                    response = requests.post(
                        url,
                        files=files,
                        timeout=self.timeout
                    )

                if response.status_code != 200:
                    error_msg = f"MinerU API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = f"{error_msg} - {error_data.get('error', 'Unknown error')}"
                    except:
                        error_msg = f"{error_msg} - {response.text}"

                    raise Exception(error_msg)

                result = response.json()

                # Добавляем метаданные
                result["_metadata"] = {
                    "file_hash": file_hash,
                    "filename": processed_filename,
                    "file_size": len(file_bytes),
                    "analyze_type": analyze_type,
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }

                # Сохраняем в кэш
                if use_cache and result.get("status") == "success":
                    self._save_to_cache(file_hash, result)

                logger.info(
                    f"Анализ файла {processed_filename} завершен за {result['_metadata']['processing_time']:.2f} секунд")

                return result

            finally:
                # Удаляем временный файл
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Ошибка анализа документа: {e}")
            return {
                "status": "error",
                "error": str(e),
                "_metadata": {
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            }

    def analyze_pdf(
            self,
            pdf_data: Union[bytes, str, Path],
            filename: Optional[str] = None,
            page_limit: int = 10,
            use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Анализ PDF документа через MinerU

        Args:
            pdf_data: Данные PDF файла
            filename: Имя файла
            page_limit: Ограничение по количеству страниц для обработки
            use_cache: Использовать кэш

        Returns:
            Результат анализа PDF
        """
        start_time = time.time()

        try:
            # Подготавливаем файл
            file_bytes, processed_filename = self._prepare_file_for_upload(pdf_data, filename)

            # Проверяем, что это PDF
            if not processed_filename.lower().endswith('.pdf'):
                if isinstance(pdf_data, bytes) and pdf_data[:4] == b'%PDF':
                    processed_filename = f"{processed_filename}.pdf"
                else:
                    raise ValueError("Файл не является PDF")

            # Вычисляем хэш
            file_hash = self.calculate_file_hash(file_bytes)

            # Проверяем кэш
            if use_cache:
                cached_result = self._check_cache(file_hash)
                if cached_result:
                    cached_result["_cached"] = True
                    cached_result["_processing_time"] = time.time() - start_time
                    return cached_result

            # Отправляем запрос в MinerU для извлечения изображений из PDF
            url = f"{self.base_url}/extract/images-from-pdf"

            # Создаем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                logger.info(f"Анализ PDF {processed_filename} через MinerU")

                with open(tmp_path, 'rb') as f:
                    files = {'file': (processed_filename, f)}
                    data = {'page_limit': str(page_limit)}

                    response = requests.post(
                        url,
                        files=files,
                        data=data,
                        timeout=self.timeout
                    )

                if response.status_code != 200:
                    error_msg = f"MinerU PDF API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = f"{error_msg} - {error_data.get('error', 'Unknown error')}"
                    except:
                        error_msg = f"{error_msg} - {response.text}"

                    raise Exception(error_msg)

                result = response.json()

                # Добавляем метаданные
                result["_metadata"] = {
                    "file_hash": file_hash,
                    "filename": processed_filename,
                    "file_size": len(file_bytes),
                    "page_limit": page_limit,
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }

                # Сохраняем в кэш
                if use_cache and result.get("status") == "success":
                    self._save_to_cache(file_hash, result)

                logger.info(
                    f"Анализ PDF {processed_filename} завершен за {result['_metadata']['processing_time']:.2f} секунд")

                return result

            finally:
                # Удаляем временный файл
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Ошибка анализа PDF: {e}")
            return {
                "status": "error",
                "error": str(e),
                "_metadata": {
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            }

    def batch_analyze_documents(
            self,
            documents: List[Dict[str, Union[bytes, str, Path]]],
            analyze_type: str = "full",
            max_workers: int = 3
    ) -> Dict[str, Any]:
        """
        Пакетный анализ нескольких документов

        Args:
            documents: Список словарей с ключами 'data' и 'filename'
            analyze_type: Тип анализа
            max_workers: Максимальное количество одновременных запросов

        Returns:
            Результаты пакетного анализа
        """
        start_time = time.time()
        results = []
        successful = 0
        failed = 0

        try:
            # Ограничиваем количество одновременных запросов
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Создаем задачи для каждого документа
                future_to_doc = {}
                for i, doc in enumerate(documents):
                    data = doc.get('data')
                    filename = doc.get('filename', f"document_{i + 1}")

                    future = executor.submit(
                        self.analyze_document,
                        data,
                        filename,
                        use_cache=True,
                        analyze_type=analyze_type
                    )
                    future_to_doc[future] = {"index": i, "filename": filename}

                # Обрабатываем результаты
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc_info = future_to_doc[future]
                    try:
                        result = future.result()
                        results.append({
                            "filename": doc_info["filename"],
                            "result": result,
                            "status": "success" if result.get("status") == "success" else "failed"
                        })
                        if result.get("status") == "success":
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        results.append({
                            "filename": doc_info["filename"],
                            "error": str(e),
                            "status": "failed"
                        })
                        failed += 1

            processing_time = time.time() - start_time

            return {
                "status": "success",
                "total_documents": len(documents),
                "successful": successful,
                "failed": failed,
                "processing_time": processing_time,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Ошибка пакетного анализа: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_documents": len(documents),
                "successful": successful,
                "failed": failed,
                "processing_time": time.time() - start_time,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

    def extract_text_from_result(self, analysis_result: Dict[str, Any]) -> str:
        """
        Извлечение текста из результата анализа MinerU

        Args:
            analysis_result: Результат анализа от MinerU

        Returns:
            Извлеченный текст
        """
        try:
            if analysis_result.get("status") != "success":
                return ""

            # Извлекаем текст из разных форматов ответа MinerU
            text = ""

            # Формат 1: Прямое поле text
            if "text" in analysis_result:
                text = analysis_result["text"]

            # Формат 2: Структурированные данные
            elif "structured_data" in analysis_result:
                structured = analysis_result["structured_data"]

                # Извлекаем текст из text_blocks
                if "text_blocks" in structured:
                    text_parts = []
                    for block in structured["text_blocks"]:
                        if "text" in block:
                            text_parts.append(block["text"])
                    text = "\n\n".join(text_parts)

                # Извлекаем заголовки
                if "headers" in structured:
                    headers = []
                    for header in structured["headers"]:
                        if "text" in header:
                            headers.append(header["text"])
                    if headers:
                        text = "\n".join(headers) + "\n\n" + text

            # Формат 3: Страницы (для PDF)
            elif "pages" in analysis_result:
                pages_text = []
                for page in analysis_result["pages"]:
                    page_text = self.extract_text_from_result(page)
                    if page_text:
                        pages_text.append(f"--- Страница {page.get('page_number', '?')} ---\n{page_text}")
                text = "\n\n".join(pages_text)

            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка извлечения текста: {e}")
            return ""

    def extract_tables_from_result(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Извлечение таблиц из результата анализа MinerU

        Args:
            analysis_result: Результат анализа от MinerU

        Returns:
            Список таблиц
        """
        try:
            if analysis_result.get("status") != "success":
                return []

            tables = []

            # Формат 1: Прямое поле tables
            if "tables" in analysis_result:
                tables_data = analysis_result["tables"]
                if isinstance(tables_data, dict) and "tables" in tables_data:
                    tables.extend(tables_data["tables"])
                elif isinstance(tables_data, list):
                    tables.extend(tables_data)

            # Формат 2: Структурированные данные
            elif "structured_data" in analysis_result:
                structured = analysis_result["structured_data"]
                if "tables" in structured:
                    tables.extend(structured["tables"])

            # Формат 3: Страницы (для PDF)
            elif "pages" in analysis_result:
                for page in analysis_result["pages"]:
                    page_tables = self.extract_tables_from_result(page)
                    for table in page_tables:
                        table["page"] = page.get("page_number", "unknown")
                    tables.extend(page_tables)

            return tables

        except Exception as e:
            logger.error(f"Ошибка извлечения таблиц: {e}")
            return []

    def extract_images_from_result(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Извлечение изображений из результата анализа MinerU

        Args:
            analysis_result: Результат анализа от MinerU

        Returns:
            Список изображений
        """
        try:
            if analysis_result.get("status") != "success":
                return []

            images = []

            # Формат 1: Прямое поле images
            if "images" in analysis_result:
                images_data = analysis_result["images"]
                if isinstance(images_data, dict) and "images" in images_data:
                    images.extend(images_data["images"])
                elif isinstance(images_data, list):
                    images.extend(images_data)

            # Формат 2: Структурированные данные
            elif "structured_data" in analysis_result:
                structured = analysis_result["structured_data"]
                if "images" in structured:
                    images.extend(structured["images"])
                if "figures" in structured:
                    images.extend(structured["figures"])

            # Формат 3: Страницы (для PDF)
            elif "pages" in analysis_result:
                for page in analysis_result["pages"]:
                    page_images = self.extract_images_from_result(page)
                    for img in page_images:
                        img["page"] = page.get("page_number", "unknown")
                    images.extend(page_images)

            return images

        except Exception as e:
            logger.error(f"Ошибка извлечения изображений: {e}")
            return []

    def get_image_preview(
            self,
            image_data: Union[str, bytes, Dict[str, Any]],
            max_width: int = 800,
            max_height: int = 600
    ) -> Dict[str, Any]:
        """
        Получение превью изображения

        Args:
            image_data: Данные изображения (base64, bytes или словарь с minio_path)
            max_width: Максимальная ширина превью
            max_height: Максимальная высота превью

        Returns:
            Информация о превью изображения
        """
        try:
            # Если это словарь с minio_path, получаем изображение из MinerU
            if isinstance(image_data, dict) and "minio_path" in image_data:
                minio_path = image_data["minio_path"]
                url = f"{self.base_url}/images/{minio_path}/preview"

                response = requests.get(
                    url,
                    params={"width": max_width, "height": max_height},
                    timeout=30
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Ошибка получения превью: HTTP {response.status_code}")

            # Если это base64 строка
            elif isinstance(image_data, str) and image_data.startswith('data:'):
                # Декодируем base64
                if ';base64,' in image_data:
                    header, data = image_data.split(';base64,', 1)
                    img_bytes = base64.b64decode(data)

                    # Создаем превью
                    return self._create_preview_from_bytes(img_bytes, max_width, max_height)
                else:
                    raise ValueError("Некорректный формат base64")

            # Если это bytes
            elif isinstance(image_data, bytes):
                return self._create_preview_from_bytes(image_data, max_width, max_height)

            else:
                raise TypeError(f"Неподдерживаемый тип данных изображения: {type(image_data)}")

        except Exception as e:
            logger.error(f"Ошибка получения превью изображения: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _create_preview_from_bytes(
            self,
            image_bytes: bytes,
            max_width: int,
            max_height: int
    ) -> Dict[str, Any]:
        """Создание превью из байтов изображения"""
        try:
            # Загружаем изображение
            img = Image.open(io.BytesIO(image_bytes))
            original_size = img.size

            # Создаем превью
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            # Конвертируем в base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            preview_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return {
                "status": "success",
                "preview_base64": f"data:image/jpeg;base64,{preview_base64}",
                "preview_size": img.size,
                "original_size": original_size,
                "format": img.format
            }

        except Exception as e:
            raise Exception(f"Ошибка создания превью: {e}")

    def clear_cache(self) -> Dict[str, Any]:
        """Очистка кэша сервиса"""
        cache_size = len(self._cache)
        self._cache.clear()

        return {
            "status": "success",
            "message": f"Кэш очищен, удалено {cache_size} элементов",
            "timestamp": datetime.now().isoformat()
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        now = time.time()
        cache_items = []

        for file_hash, cached_item in self._cache.items():
            age = now - cached_item["timestamp"]
            cache_items.append({
                "file_hash": file_hash[:16] + "...",  # Сокращаем для безопасности
                "age_seconds": age,
                "age_human": f"{age:.1f}s"
            })

        return {
            "status": "success",
            "cache_size": len(self._cache),
            "cache_ttl": self._cache_ttl,
            "items": cache_items,
            "timestamp": datetime.now().isoformat()
        }


# Создаем синглтон экземпляр сервиса
mineru_service = MinerUService()


# Утилитарные функции
def convert_image_to_base64(image_path: Union[str, Path], format: str = "JPEG") -> str:
    """
    Конвертация изображения в base64 строку

    Args:
        image_path: Путь к изображению
        format: Формат изображения (JPEG, PNG, etc.)

    Returns:
        Base64 строка с изображением
    """
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()

        img_str = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"

    except Exception as e:
        logger.error(f"Ошибка конвертации изображения в base64: {e}")
        return ""


def validate_mineru_response(response: Dict[str, Any]) -> bool:
    """
    Валидация ответа MinerU

    Args:
        response: Ответ от MinerU API

    Returns:
        True если ответ валиден
    """
    if not isinstance(response, dict):
        return False

    if "status" not in response:
        return False

    if response["status"] == "success":
        # Для успешного ответа проверяем наличие основных полей
        required_fields = ["_metadata"]
        for field in required_fields:
            if field not in response:
                logger.warning(f"В успешном ответе MinerU отсутствует поле: {field}")
                return False

    return True


# Декоратор для обработки ошибок MinerU
def handle_mineru_errors(func):
    """Декоратор для обработки ошибок MinerU"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"MinerU network error: {e}")
            return {
                "status": "error",
                "error": f"Сетевая ошибка MinerU: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"MinerU service error: {e}")
            return {
                "status": "error",
                "error": f"Ошибка сервиса MinerU: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    return wrapper


# Пример использования сервиса
if __name__ == "__main__":
    # Инициализация сервиса
    service = MinerUService()

    # Проверка здоровья
    health = service.health_check()
    print(f"Health check: {health}")

    # Пример анализа локального PDF файла
    if os.path.exists("test.pdf"):
        result = service.analyze_pdf("test.pdf")
        print(f"PDF analysis result status: {result.get('status')}")

        # Извлечение текста
        text = service.extract_text_from_result(result)
        print(f"Extracted text length: {len(text)} characters")

        # Извлечение таблиц
        tables = service.extract_tables_from_result(result)
        print(f"Extracted tables count: {len(tables)}")

        # Извлечение изображений
        images = service.extract_images_from_result(result)
        print(f"Extracted images count: {len(images)}")

    # Статистика кэша
    cache_stats = service.get_cache_stats()
    print(f"Cache stats: {cache_stats}")