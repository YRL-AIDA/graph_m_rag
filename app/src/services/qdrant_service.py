# src/services/qdrant_service.py
import uuid
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import logging

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    CollectionDescription,
    ScoredPoint,
    UpdateResult,
    SearchRequest,
    RecommendRequest,
    CountRequest,
    CountResult,
    SnapshotDescription,
)

from app.config import settings


class VectorDistance(str, Enum):
    """Метрики расстояния для векторов"""
    COSINE = "Cosine"
    EUCLIDEAN = "Euclidean"
    DOT = "Dot"


class CollectionStatus(str, Enum):
    """Статусы коллекции"""
    GREEN = "green"  # Все сегменты готовы
    YELLOW = "yellow"  # Некоторые сегменты могут быть не готовы
    RED = "red"  # Коллекция не доступна


@dataclass
class SearchResult:
    """Результат поиска"""
    id: Union[str, int]
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        result = {
            "id": self.id,
            "score": self.score,
            "payload": self.payload
        }
        if self.vector:
            result["vector"] = self.vector
        return result


@dataclass
class CollectionInfo:
    """Информация о коллекции"""
    name: str
    status: CollectionStatus
    vectors_count: int
    points_count: int
    segments_count: int
    config: Dict[str, Any]
    payload_schema: Dict[str, Any]
    created_at: datetime

    @classmethod
    def from_qdrant_response(cls, response: CollectionDescription) -> 'CollectionInfo':
        """Создание из ответа Qdrant"""
        return cls(
            name=response.name,
            status=CollectionStatus(response.status),
            vectors_count=response.vectors_count,
            points_count=response.points_count,
            segments_count=response.segments_count,
            config=response.config.dict() if response.config else {},
            payload_schema=response.payload_schema.dict() if response.payload_schema else {},
            created_at=datetime.now()  # Qdrant не возвращает время создания
        )


@dataclass
class PointData:
    """Данные точки"""
    id: Union[str, int]
    vector: List[float]
    payload: Dict[str, Any]

    def to_point_struct(self) -> PointStruct:
        """Конвертация в PointStruct Qdrant"""
        return PointStruct(
            id=self.id,
            vector=self.vector,
            payload=self.payload
        )


class QdrantService:
    """Сервис для работы с Qdrant (векторная база данных)"""

    def __init__(self, use_async: bool = False):
        """
        Инициализация клиента Qdrant

        Args:
            use_async: Использовать асинхронный клиент
        """
        self.config = settings.qdrant
        self.use_async = use_async
        self.logger = logging.getLogger(__name__)

        # Инициализация клиента
        if use_async:
            self.client = AsyncQdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            self.sync_client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        else:
            self.client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            self.async_client = None

        # Основная коллекция
        self.collection_name = self.config.collection_name
        self.vector_size = self.config.vector_size
        self.distance = self._get_distance(self.config.distance)

        # Кэш информации о коллекциях
        self._collection_cache = {}
        self._cache_ttl = 60  # секунды
        self._last_cache_update = 0

    def _get_distance(self, distance_str: str) -> Distance:
        """Получение метрики расстояния"""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        return distance_map.get(distance_str.lower(), Distance.COSINE)

    async def _get_async_client(self) -> AsyncQdrantClient:
        """Получение асинхронного клиента"""
        if not hasattr(self, 'async_client') or self.async_client is None:
            self.async_client = AsyncQdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        return self.async_client

    def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья Qdrant"""
        try:
            # Пытаемся получить список коллекций
            collections = self.client.get_collections()

            # Проверяем подключение к основной коллекции
            collection_exists = self.collection_exists(self.collection_name)

            # Получаем информацию о сервере
            try:
                from qdrant_client.http.api_client import ApiClient
                # Это внутренний API, может измениться
                # В реальном проекте используйте официальные методы
                pass
            except:
                pass

            return {
                "status": "healthy",
                "collections_count": len(collections.collections),
                "main_collection_exists": collection_exists,
                "qdrant_url": self.config.url,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Qdrant health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def collection_exists(self, collection_name: str) -> bool:
        """Проверка существования коллекции"""
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False

    def create_collection(
            self,
            collection_name: Optional[str] = None,
            vector_size: Optional[int] = None,
            distance: Optional[Distance] = None,
            **kwargs
    ) -> bool:
        """
        Создание коллекции

        Args:
            collection_name: Имя коллекции (если None, используется основная)
            vector_size: Размерность векторов
            distance: Метрика расстояния
            **kwargs: Дополнительные параметры для коллекции

        Returns:
            True если коллекция создана успешно
        """
        try:
            name = collection_name or self.collection_name
            size = vector_size or self.vector_size
            dist = distance or self.distance

            # Дополнительные параметры
            hnsw_config = kwargs.get("hnsw_config", None)
            quantization_config = kwargs.get("quantization_config", None)
            on_disk_payload = kwargs.get("on_disk_payload", True)

            # Создаем коллекцию
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=size,
                    distance=dist
                ),
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
                on_disk_payload=on_disk_payload
            )

            self.logger.info(f"Collection '{name}' created successfully")

            # Сбрасываем кэш
            self._collection_cache.pop(name, None)

            return True

        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            # Если коллекция уже существует, это не ошибка
            if "already exists" in str(e):
                return True
            return False

    def get_collection_info(self, collection_name: Optional[str] = None) -> Optional[CollectionInfo]:
        """Получение информации о коллекции"""
        try:
            name = collection_name or self.collection_name

            # Проверяем кэш
            current_time = time.time()
            if (name in self._collection_cache and
                    current_time - self._last_cache_update < self._cache_ttl):
                return self._collection_cache[name]

            # Получаем информацию из Qdrant
            collection_info = self.client.get_collection(collection_name=name)

            # Преобразуем в наш формат
            info = CollectionInfo.from_qdrant_response(collection_info)

            # Обновляем кэш
            self._collection_cache[name] = info
            self._last_cache_update = current_time

            return info

        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return None

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Удаление коллекции"""
        try:
            name = collection_name or self.collection_name

            if not self.collection_exists(name):
                self.logger.warning(f"Collection '{name}' does not exist")
                return True

            self.client.delete_collection(collection_name=name)

            # Сбрасываем кэш
            self._collection_cache.pop(name, None)

            self.logger.info(f"Collection '{name}' deleted successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            return False

    def upsert_points(
            self,
            points: List[PointData],
            collection_name: Optional[str] = None,
            wait: bool = True,
            **kwargs
    ) -> Optional[UpdateResult]:
        """
        Добавление или обновление точек

        Args:
            points: Список точек для добавления
            collection_name: Имя коллекции
            wait: Ждать подтверждения записи
            **kwargs: Дополнительные параметры

        Returns:
            Результат обновления или None при ошибке
        """
        try:
            name = collection_name or self.collection_name

            # Проверяем существование коллекции
            if not self.collection_exists(name):
                self.logger.warning(f"Collection '{name}' does not exist, creating...")
                if not self.create_collection(name):
                    raise Exception(f"Failed to create collection '{name}'")

            # Конвертируем точки в формат Qdrant
            qdrant_points = [p.to_point_struct() for p in points]

            # Выполняем upsert
            result = self.client.upsert(
                collection_name=name,
                points=qdrant_points,
                wait=wait,
                **kwargs
            )

            self.logger.info(f"Upserted {len(points)} points to collection '{name}'")

            # Сбрасываем кэш для этой коллекции
            self._collection_cache.pop(name, None)

            return result

        except Exception as e:
            self.logger.error(f"Error upserting points: {e}")
            return None

    def upsert_embeddings(
            self,
            embeddings: List[List[float]],
            payloads: List[Dict[str, Any]],
            ids: Optional[List[Union[str, int]]] = None,
            collection_name: Optional[str] = None,
            **kwargs
    ) -> Optional[UpdateResult]:
        """
        Добавление эмбеддингов с метаданными

        Args:
            embeddings: Список векторов
            payloads: Список метаданных
            ids: Список ID (если None, генерируются автоматически)
            collection_name: Имя коллекции
            **kwargs: Дополнительные параметры

        Returns:
            Результат обновления
        """
        try:
            if len(embeddings) != len(payloads):
                raise ValueError("Number of embeddings must match number of payloads")

            if ids is None:
                # Генерируем UUID для каждой точки
                ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            elif len(ids) != len(embeddings):
                raise ValueError("Number of IDs must match number of embeddings")

            # Создаем точки
            points = []
            for i, (embedding, payload, point_id) in enumerate(zip(embeddings, payloads, ids)):
                # Добавляем timestamp в payload
                enhanced_payload = payload.copy()
                if "timestamp" not in enhanced_payload:
                    enhanced_payload["timestamp"] = datetime.now().isoformat()

                # Добавляем индекс в payload для отслеживания
                enhanced_payload["batch_index"] = i

                points.append(PointData(
                    id=point_id,
                    vector=embedding,
                    payload=enhanced_payload
                ))

            # Выполняем upsert
            return self.upsert_points(
                points=points,
                collection_name=collection_name,
                **kwargs
            )

        except Exception as e:
            self.logger.error(f"Error upserting embeddings: {e}")
            return None

    def search(
            self,
            query_vector: List[float],
            collection_name: Optional[str] = None,
            limit: int = 10,
            score_threshold: Optional[float] = None,
            filter_condition: Optional[Filter] = None,
            with_payload: bool = True,
            with_vectors: bool = False,
            **kwargs
    ) -> List[SearchResult]:
        """
        Поиск похожих векторов

        Args:
            query_vector: Вектор запроса
            collection_name: Имя коллекции
            limit: Максимальное количество результатов
            score_threshold: Порог схожести
            filter_condition: Фильтр по метаданным
            with_payload: Включать метаданные в результат
            with_vectors: Включать векторы в результат
            **kwargs: Дополнительные параметры

        Returns:
            Список результатов поиска
        """
        try:
            name = collection_name or self.collection_name

            # Выполняем поиск
            search_result = self.client.search(
                collection_name=name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_condition,
                with_payload=with_payload,
                with_vectors=with_vectors,
                **kwargs
            )

            # Конвертируем в наш формат
            results = []
            for point in search_result:
                results.append(SearchResult(
                    id=point.id,
                    score=point.score,
                    payload=point.payload or {},
                    vector=point.vector if with_vectors else None
                ))

            return results

        except Exception as e:
            self.logger.error(f"Error searching vectors: {e}")
            return []

    def search_by_text(
            self,
            query_text: str,
            text_field: str = "text",
            collection_name: Optional[str] = None,
            limit: int = 10,
            **kwargs
    ) -> List[SearchResult]:
        """
        Поиск по текстовому полю с использованием фильтра

        Args:
            query_text: Текст для поиска
            text_field: Поле с текстом в payload
            collection_name: Имя коллекции
            limit: Максимальное количество результатов
            **kwargs: Дополнительные параметры

        Returns:
            Список результатов поиска
        """
        try:
            # Создаем фильтр для текстового поиска
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key=text_field,
                        match=MatchValue(value=query_text)
                    )
                ]
            )

            # Для текстового поиска без вектора ищем все точки с фильтром
            # Сначала получаем количество точек
            count_result = self.count_points(
                collection_name=collection_name,
                filter_condition=filter_condition
            )

            if count_result.count == 0:
                return []

            # Получаем все точки с фильтром
            points = self.get_points(
                collection_name=collection_name,
                filter_condition=filter_condition,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            # Конвертируем в результаты поиска
            results = []
            for point in points:
                results.append(SearchResult(
                    id=point.id,
                    score=1.0,  # Для текстового поиска без вектора score не имеет смысла
                    payload=point.payload or {}
                ))

            return results

        except Exception as e:
            self.logger.error(f"Error in text search: {e}")
            return []

    def hybrid_search(
            self,
            query_vector: List[float],
            query_text: Optional[str] = None,
            text_field: str = "text",
            collection_name: Optional[str] = None,
            vector_weight: float = 0.7,
            text_weight: float = 0.3,
            limit: int = 10,
            **kwargs
    ) -> List[SearchResult]:
        """
        Гибридный поиск (векторный + текстовый)

        Args:
            query_vector: Вектор запроса
            query_text: Текст запроса
            text_field: Поле с текстом в payload
            collection_name: Имя коллекции
            vector_weight: Вес векторного поиска
            text_weight: Вес текстового поиска
            limit: Максимальное количество результатов
            **kwargs: Дополнительные параметры

        Returns:
            Список результатов с комбинированным score
        """
        try:
            name = collection_name or self.collection_name

            # Выполняем векторный поиск
            vector_results = self.search(
                query_vector=query_vector,
                collection_name=name,
                limit=limit * 2,  # Берем больше для комбинирования
                with_payload=True,
                with_vectors=False
            )

            # Если есть текст запроса, выполняем текстовый поиск
            text_results = []
            if query_text:
                text_results = self.search_by_text(
                    query_text=query_text,
                    text_field=text_field,
                    collection_name=name,
                    limit=limit * 2
                )

            # Объединяем результаты
            all_results = {}

            # Добавляем векторные результаты
            for result in vector_results:
                if result.id not in all_results:
                    all_results[result.id] = {
                        "id": result.id,
                        "vector_score": result.score,
                        "text_score": 0.0,
                        "payload": result.payload
                    }
                else:
                    all_results[result.id]["vector_score"] = result.score

            # Добавляем текстовые результаты
            for result in text_results:
                if result.id not in all_results:
                    all_results[result.id] = {
                        "id": result.id,
                        "vector_score": 0.0,
                        "text_score": 1.0,  # Для точного текстового совпадения
                        "payload": result.payload
                    }
                else:
                    all_results[result.id]["text_score"] = 1.0

            # Вычисляем комбинированный score
            combined_results = []
            for result_data in all_results.values():
                combined_score = (
                        result_data["vector_score"] * vector_weight +
                        result_data["text_score"] * text_weight
                )

                combined_results.append(SearchResult(
                    id=result_data["id"],
                    score=combined_score,
                    payload=result_data["payload"]
                ))

            # Сортируем по score и берем top-k
            combined_results.sort(key=lambda x: x.score, reverse=True)

            return combined_results[:limit]

        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            # Fallback к векторному поиску
            return self.search(
                query_vector=query_vector,
                collection_name=collection_name,
                limit=limit,
                **kwargs
            )

    def get_point(self, point_id: Union[str, int], collection_name: Optional[str] = None) -> Optional[PointData]:
        """Получение точки по ID"""
        try:
            name = collection_name or self.collection_name

            result = self.client.retrieve(
                collection_name=name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )

            if not result:
                return None

            point = result[0]
            return PointData(
                id=point.id,
                vector=point.vector,
                payload=point.payload or {}
            )

        except Exception as e:
            self.logger.error(f"Error getting point: {e}")
            return None

    def get_points(
            self,
            collection_name: Optional[str] = None,
            ids: Optional[List[Union[str, int]]] = None,
            filter_condition: Optional[Filter] = None,
            limit: int = 100,
            offset: int = 0,
            with_payload: bool = True,
            with_vectors: bool = False,
            **kwargs
    ) -> List[PointData]:
        """Получение точек с фильтрацией"""
        try:
            name = collection_name or self.collection_name

            if ids:
                # Получаем точки по ID
                result = self.client.retrieve(
                    collection_name=name,
                    ids=ids,
                    with_payload=with_payload,
                    with_vectors=with_vectors
                )

                points = []
                for point in result:
                    points.append(PointData(
                        id=point.id,
                        vector=point.vector if with_vectors else [],
                        payload=point.payload or {}
                    ))

                return points
            else:
                # Используем scroll для получения точек с фильтром
                scroll_result = self.client.scroll(
                    collection_name=name,
                    scroll_filter=filter_condition,
                    limit=limit,
                    offset=offset,
                    with_payload=with_payload,
                    with_vectors=with_vectors
                )

                points = []
                for point in scroll_result[0]:
                    points.append(PointData(
                        id=point.id,
                        vector=point.vector if with_vectors else [],
                        payload=point.payload or {}
                    ))

                return points

        except Exception as e:
            self.logger.error(f"Error getting points: {e}")
            return []

    def update_point_payload(
            self,
            point_id: Union[str, int],
            payload: Dict[str, Any],
            collection_name: Optional[str] = None,
            **kwargs
    ) -> bool:
        """Обновление метаданных точки"""
        try:
            name = collection_name or self.collection_name

            # Получаем текущие payload
            current_point = self.get_point(point_id, name)
            if not current_point:
                return False

            # Обновляем payload
            updated_payload = current_point.payload.copy()
            updated_payload.update(payload)

            # Устанавливаем обновленный payload
            self.client.set_payload(
                collection_name=name,
                payload=updated_payload,
                points=[point_id],
                **kwargs
            )

            self.logger.info(f"Updated payload for point {point_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating point payload: {e}")
            return False

    def delete_points(
            self,
            point_ids: List[Union[str, int]],
            collection_name: Optional[str] = None,
            **kwargs
    ) -> bool:
        """Удаление точек по ID"""
        try:
            name = collection_name or self.collection_name

            self.client.delete(
                collection_name=name,
                points_selector=models.PointIdsList(
                    points=point_ids
                ),
                **kwargs
            )

            self.logger.info(f"Deleted {len(point_ids)} points from collection '{name}'")

            # Сбрасываем кэш
            self._collection_cache.pop(name, None)

            return True

        except Exception as e:
            self.logger.error(f"Error deleting points: {e}")
            return False

    def delete_points_by_filter(
            self,
            filter_condition: Filter,
            collection_name: Optional[str] = None,
            **kwargs
    ) -> bool:
        """Удаление точек по фильтру"""
        try:
            name = collection_name or self.collection_name

            self.client.delete(
                collection_name=name,
                points_selector=models.FilterSelector(
                    filter=filter_condition
                ),
                **kwargs
            )

            self.logger.info(f"Deleted points by filter from collection '{name}'")

            # Сбрасываем кэш
            self._collection_cache.pop(name, None)

            return True

        except Exception as e:
            self.logger.error(f"Error deleting points by filter: {e}")
            return False

    def count_points(
            self,
            collection_name: Optional[str] = None,
            filter_condition: Optional[Filter] = None,
            **kwargs
    ) -> CountResult:
        """Подсчет точек в коллекции"""
        try:
            name = collection_name or self.collection_name

            count_request = CountRequest(
                filter=filter_condition,
                **kwargs
            )

            return self.client.count(
                collection_name=name,
                count_request=count_request
            )

        except Exception as e:
            self.logger.error(f"Error counting points: {e}")
            return CountResult(count=0)

    def create_snapshot(
            self,
            collection_name: Optional[str] = None,
            wait: bool = True
    ) -> Optional[SnapshotDescription]:
        """Создание снапшота коллекции"""
        try:
            name = collection_name or self.collection_name

            return self.client.create_snapshot(
                collection_name=name,
                wait=wait
            )

        except Exception as e:
            self.logger.error(f"Error creating snapshot: {e}")
            return None

    def list_snapshots(
            self,
            collection_name: Optional[str] = None
    ) -> List[SnapshotDescription]:
        """Получение списка снапшотов"""
        try:
            name = collection_name or self.collection_name

            result = self.client.list_snapshots(
                collection_name=name
            )

            return result

        except Exception as e:
            self.logger.error(f"Error listing snapshots: {e}")
            return []

    def recreate_collection_from_snapshot(
            self,
            snapshot_name: str,
            collection_name: Optional[str] = None,
            new_collection_name: Optional[str] = None
    ) -> bool:
        """Восстановление коллекции из снапшота"""
        try:
            name = collection_name or self.collection_name
            new_name = new_collection_name or f"{name}_restored"

            # Удаляем коллекцию если существует
            if self.collection_exists(new_name):
                self.delete_collection(new_name)

            # Восстанавливаем из снапшота
            self.client.recover_snapshot(
                collection_name=new_name,
                location=snapshot_name,
                wait=True
            )

            self.logger.info(f"Restored collection '{new_name}' from snapshot '{snapshot_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error restoring from snapshot: {e}")
            return False

    def batch_operations(
            self,
            operations: List[Dict[str, Any]],
            collection_name: Optional[str] = None
    ) -> bool:
        """Выполнение пакетных операций"""
        try:
            name = collection_name or self.collection_name

            # Конвертируем операции в формат Qdrant
            qdrant_operations = []
            for op in operations:
                op_type = op.get("type")

                if op_type == "upsert":
                    points = [PointData(**p).to_point_struct() for p in op.get("points", [])]
                    qdrant_operations.append(
                        models.UpsertOperation(
                            upsert=models.PointsList(
                                points=points
                            )
                        )
                    )
                elif op_type == "delete":
                    qdrant_operations.append(
                        models.DeleteOperation(
                            delete=models.PointsSelector(
                                points=models.PointIdsList(
                                    points=op.get("ids", [])
                                )
                            )
                        )
                    )
                elif op_type == "set_payload":
                    qdrant_operations.append(
                        models.SetPayloadOperation(
                            set_payload=models.SetPayload(
                                payload=op.get("payload", {}),
                                points=op.get("ids", [])
                            )
                        )
                    )

            # Выполняем пакетные операции
            self.client.batch_update_points(
                collection_name=name,
                update_operations=qdrant_operations
            )

            self.logger.info(f"Executed {len(operations)} batch operations on collection '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error executing batch operations: {e}")
            return False

    def create_payload_index(
            self,
            field_name: str,
            field_schema: Optional[Dict[str, Any]] = None,
            collection_name: Optional[str] = None
    ) -> bool:
        """Создание индекса для поля в payload"""
        try:
            name = collection_name or self.collection_name

            if field_schema is None:
                field_schema = {"type": "keyword"}

            self.client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=field_schema
            )

            self.logger.info(f"Created payload index for field '{field_name}' in collection '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error creating payload index: {e}")
            return False

    def delete_payload_index(
            self,
            field_name: str,
            collection_name: Optional[str] = None
    ) -> bool:
        """Удаление индекса для поля в payload"""
        try:
            name = collection_name or self.collection_name

            self.client.delete_payload_index(
                collection_name=name,
                field_name=field_name
            )

            self.logger.info(f"Deleted payload index for field '{field_name}' in collection '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting payload index: {e}")
            return False

    def optimize_collection(
            self,
            collection_name: Optional[str] = None,
            **kwargs
    ) -> bool:
        """Оптимизация коллекции"""
        try:
            name = collection_name or self.collection_name

            # Параметры оптимизации по умолчанию
            default_params = {
                "max_segment_size": None,
                "memmap_threshold": None,
                "indexing_threshold": None,
                "flush_interval_sec": None,
                "vacuum": True,
                "optimizers_config": None
            }
            default_params.update(kwargs)

            self.client.update_collection(
                collection_name=name,
                optimizer_config=default_params.get("optimizers_config")
            )

            # Если указан vacuum, выполняем его
            if default_params.get("vacuum", True):
                self.client.force_vacuum(
                    collection_name=name
                )

            self.logger.info(f"Optimized collection '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error optimizing collection: {e}")
            return False

    def get_collection_stats(
            self,
            collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Получение статистики коллекции"""
        try:
            name = collection_name or self.collection_name

            collection_info = self.get_collection_info(name)
            if not collection_info:
                return {}

            # Подсчитываем точки с разными фильтрами
            total_count = self.count_points(name)

            # Получаем информацию о сегментах
            cluster_info = self.client.cluster_status()

            stats = {
                "collection_name": name,
                "status": collection_info.status.value,
                "total_points": total_count.count,
                "vectors_count": collection_info.vectors_count,
                "segments_count": collection_info.segments_count,
                "cluster_status": cluster_info.status if cluster_info else "unknown",
                "timestamp": datetime.now().isoformat(),
                "payload_fields": list(collection_info.payload_schema.keys()) if collection_info.payload_schema else []
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {}

    def clear_collection(
            self,
            collection_name: Optional[str] = None,
            keep_collection: bool = True
    ) -> bool:
        """Очистка коллекции (удаление всех точек)"""
        try:
            name = collection_name or self.collection_name

            if not self.collection_exists(name):
                return True

            if keep_collection:
                # Удаляем все точки
                self.delete_points_by_filter(
                    filter_condition=Filter(),
                    collection_name=name
                )
                self.logger.info(f"Cleared all points from collection '{name}'")
            else:
                # Удаляем всю коллекцию
                self.delete_collection(name)
                self.logger.info(f"Deleted collection '{name}'")

            return True

        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            return False


# Асинхронная версия сервиса
class AsyncQdrantService(QdrantService):
    """Асинхронная версия сервиса Qdrant"""

    def __init__(self):
        super().__init__(use_async=True)

    async def upsert_points_async(
            self,
            points: List[PointData],
            collection_name: Optional[str] = None,
            wait: bool = True,
            **kwargs
    ) -> Optional[UpdateResult]:
        """Асинхронное добавление точек"""
        try:
            name = collection_name or self.collection_name
            client = await self._get_async_client()

            # Проверяем существование коллекции
            collections = await client.get_collections()
            collection_exists = any(c.name == name for c in collections.collections)

            if not collection_exists:
                self.logger.warning(f"Collection '{name}' does not exist, creating...")
                await client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )

            # Конвертируем точки
            qdrant_points = [p.to_point_struct() for p in points]

            # Выполняем upsert
            result = await client.upsert(
                collection_name=name,
                points=qdrant_points,
                wait=wait,
                **kwargs
            )

            self.logger.info(f"Async upserted {len(points)} points to collection '{name}'")
            return result

        except Exception as e:
            self.logger.error(f"Error in async upsert: {e}")
            return None

    async def search_async(
            self,
            query_vector: List[float],
            collection_name: Optional[str] = None,
            limit: int = 10,
            score_threshold: Optional[float] = None,
            filter_condition: Optional[Filter] = None,
            with_payload: bool = True,
            with_vectors: bool = False,
            **kwargs
    ) -> List[SearchResult]:
        """Асинхронный поиск"""
        try:
            name = collection_name or self.collection_name
            client = await self._get_async_client()

            search_result = await client.search(
                collection_name=name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_condition,
                with_payload=with_payload,
                with_vectors=with_vectors,
                **kwargs
            )

            results = []
            for point in search_result:
                results.append(SearchResult(
                    id=point.id,
                    score=point.score,
                    payload=point.payload or {},
                    vector=point.vector if with_vectors else None
                ))

            return results

        except Exception as e:
            self.logger.error(f"Error in async search: {e}")
            return []


# Фабрика для создания сервиса
def create_qdrant_service(use_async: bool = False) -> QdrantService:
    """Создание экземпляра сервиса Qdrant"""
    if use_async:
        return AsyncQdrantService()
    return QdrantService(use_async=False)


# Глобальный экземпляр для синхронного использования
qdrant_service = QdrantService()

# Пример использования
if __name__ == "__main__":
    # Инициализация сервиса
    service = QdrantService()

    # Проверка здоровья
    health = service.health_check()
    print(f"Health check: {health}")

    # Создание коллекции (если не существует)
    if not service.collection_exists():
        service.create_collection()

    # Получение информации о коллекции
    info = service.get_collection_info()
    if info:
        print(f"Collection info: {info}")

    # Пример добавления векторов
    sample_embeddings = [[0.1] * 768, [0.2] * 768]
    sample_payloads = [
        {"text": "Sample document 1", "source": "test"},
        {"text": "Sample document 2", "source": "test"}
    ]

    result = service.upsert_embeddings(
        embeddings=sample_embeddings,
        payloads=sample_payloads
    )

    if result:
        print(f"Upsert result: {result}")

    # Пример поиска
    query_vector = [0.15] * 768
    search_results = service.search(
        query_vector=query_vector,
        limit=5
    )

    print(f"Search results: {len(search_results)} found")
    for res in search_results:
        print(f"  - ID: {res.id}, Score: {res.score:.4f}")