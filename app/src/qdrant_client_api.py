import logging
import uuid
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.config.settings import settings

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    def __init__(self, collection_name: Optional[str] = None):
        if settings.qdrant.QDRANT_API_KEY:
            self.client = QdrantClient(
                host=settings.qdrant.QDRANT_HOST,
                port=settings.qdrant.QDRANT_PORT,
                grpc_port=settings.qdrant.QDRANT_GRPC_PORT,
                prefer_grpc=True
            )
        else:
            self.client = QdrantClient(
                host=settings.qdrant.QDRANT_HOST,
                port=settings.qdrant.QDRANT_PORT
            )
        self.collection_name = collection_name or settings.qdrant.QDRANT_COLLECTION_NAME

    def create_collection(
        self,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Создает коллекцию в Qdrant с указанными параметрами
        """
        try:
            # Проверяем, существует ли уже коллекция
            if self.client.collection_exists(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                return True

            # Создаем новую коллекцию
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )

            logger.info(f"Collection {self.collection_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False

    def delete_collection(self) -> bool:
        """
        Удаляет коллекцию из Qdrant
        """
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    def upload_points(self, points: List[PointStruct]) -> bool:
        """
        Загружает точки (векторы) в коллекцию
        """
        try:
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Uploaded {len(points)} points to collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error uploading points: {e}")
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_condition: Optional[models.Filter] = None
    ) -> List[models.ScoredPoint]:
        """
        Выполняет поиск по вектору в коллекции
        """
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=filter_condition
            )
            return results
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def batch_search(
        self,
        query_vectors: List[List[float]],
        limit: int = 10,
        filter_condition: Optional[models.Filter] = None
    ) -> List[List[models.ScoredPoint]]:
        """
        Выполняет пакетный поиск по нескольким векторам
        """
        try:
            searches = [
                models.SearchRequest(
                    vector=query_vector,
                    limit=limit,
                    filter=filter_condition
                )
                for query_vector in query_vectors
            ]

            results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=searches
            )
            return results
        except Exception as e:
            logger.error(f"Error during batch search: {e}")
            return []

    def count(self) -> int:
        """
        Возвращает количество точек в коллекции
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0

    def get_point(self, point_id: str) -> Optional[models.Record]:
        """
        Получает точку по ID
        """
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            return records[0] if records else None
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None

    def delete_points(self, point_ids: List[str]) -> bool:
        """
        Удаляет точки по ID
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            logger.info(f"Deleted {len(point_ids)} points from collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting points: {e}")
            return False

    def update_point(self, point_id: str, payload: Dict[str, Any]) -> bool:
        """
        Обновляет payload точки
        """
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[point_id]
            )
            return True
        except Exception as e:
            logger.error(f"Error updating point {point_id}: {e}")
            return False

    def list_collections(self) -> List[str]:
        """
        Возвращает список всех коллекций в Qdrant
        """
        try:
            collections = self.client.get_collections().collections
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error getting collections list: {e}")
            return []

    def close(self):
        """
        Закрывает соединение с Qdrant
        """
        if hasattr(self.client, 'close'):
            self.client.close()

    def save_embeddings(self, embeddings: List[List[float]], texts: List[str],
                        metadata_list: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Сохраняет эмбеддинги в векторную базу данных Qdrant

        Args:
            embeddings: Список векторов эмбеддингов
            texts: Список текстов, соответствующих эмбеддингам
            metadata_list: Опциональный список метаданных для каждого эмбеддинга

        Returns:
            bool: True если успешно сохранено, иначе False
        """
        try:
            if len(embeddings) != len(texts):
                raise ValueError("Количество эмбеддингов должно совпадать с количеством текстов")

            if metadata_list and len(metadata_list) != len(embeddings):
                raise ValueError("Количество метаданных должно совпадать с количеством эмбеддингов")

            points = []
            for i, (embedding, text) in enumerate(zip(embeddings, texts)):
                # Генерируем уникальный ID для каждой точки
                point_id = str(uuid.uuid4())

                # Подготовляем payload
                payload = {
                    "text": text,
                    "timestamp": str(uuid.uuid4())  # можно заменить на реальную дату/время
                }

                # Добавляем метаданные, если они есть
                if metadata_list:
                    payload.update(metadata_list[i])

                # Создаем структуру точки
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )

                points.append(point)

            # Загружаем точки в коллекцию
            success = self.upload_points(points)
            if success:
                logger.info(f"Successfully saved {len(embeddings)} embeddings to collection {self.collection_name}")
                return True
            else:
                logger.error("Failed to upload embeddings to Qdrant")
                return False

        except Exception as e:
            logger.error(f"Error saving embeddings to Qdrant: {e}")
            return False


def get_qdrant_client(collection_name: Optional[str] = None) -> QdrantClientWrapper:
    """
    Фабрика для создания клиента Qdrant

    Args:
        collection_name: Имя коллекции для использования. Если не указано, используется коллекция по умолчанию.
    """
    return QdrantClientWrapper(collection_name=collection_name)


# Пример использования
if __name__ == "__main__":
    # Инициализация клиента
    qdrant_client = get_qdrant_client()

    # Создание коллекции
    success = qdrant_client.create_collection(vector_size=2048)  # Пример для OpenAI embeddings

    if success:
        print("Collection created successfully!")

        # Пример добавления точек
        points = [
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3, 0.4] * 384,  # Пример вектора
                payload={"text": "Пример документа", "metadata": {"source": "example.txt"}}
            ),
            PointStruct(
                id=2,
                vector=[0.4, 0.3, 0.2, 0.1] * 384,  # Пример вектора
                payload={"text": "Еще один документ", "metadata": {"source": "example2.txt"}}
            )
        ]

        qdrant_client.upload_points(points)

        # Пример поиска
        query_vector = [0.15, 0.25, 0.35, 0.45] * 384
        results = qdrant_client.search(query_vector, limit=5)

        print(f"Found {len(results.points)} results:")
        for result in results.points:
            print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")

    # Закрытие соединения
    qdrant_client.close()