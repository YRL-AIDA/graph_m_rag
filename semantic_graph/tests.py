import sys
import logging
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import bootstrap  # noqa: F401

import unittest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
from dotenv import load_dotenv
import os
from manager import Manager, ManagerConfig, Neo4jConnection
import numpy as np

logger = logging.getLogger(__name__)

class TestGetEntityRelationships(unittest.TestCase):
    """Тесты для метода get_entity_relationships класса Manager."""

    def setUp(self):
        """Выполняется перед каждым тестом — подготовка окружения."""
        # Создаём мок-конфигурацию
        self.config = MagicMock(spec=ManagerConfig)
        self.config.uri = "neo4j://mock:7687"
        self.config.user = "test"
        self.config.password = "test"
        self.config.name_db = "test_db"

        # Мокаем всё, что требует реальной БД
        with patch.object(Neo4jConnection, '__init__', return_value=None), \
                patch.object(Manager, 'initialize_schema', return_value=None):  # ← отключаем инициализацию

            self.manager = Manager(self.config)

            # Настраиваем только то, что реально используется в тестируемом методе
            self.manager.query = MagicMock(return_value=[])  # ← мок для self.query()
            self.manager.conn = MagicMock()
    def tearDown(self):
        """Выполняется после каждого теста — очистка."""
        self.manager.close()

    def test01_get_entity_relationships_with_results(self):
        """Тест: метод возвращает DataFrame с данными при наличии связей."""
        # 🎭 Мокаем результат запроса к Neo4j
        mock_record1 = MagicMock()
        mock_record1.data.return_value = {
            "source_title": "Alice",
            "source_type": "PERSON",
            "target_title": "Bob",
            "target_type": "PERSON",
            "weight": 2.5,
            "description": "knows",
            "text_unit_ids": ["tu1", "tu2"],
            "updated_at": "2024-01-15T10:30:00",
            "created_at": "2024-01-10T08:00:00",
        }

        mock_record2 = MagicMock()
        mock_record2.data.return_value = {
            "source_title": "CompanyX",
            "source_type": "ORGANIZATION",
            "target_title": "Alice",
            "target_type": "PERSON",
            "weight": None,  # Проверяем обработку None
            "description": None,
            "text_unit_ids": None,
            "updated_at": None,
            "created_at": None,
        }

        # Настраиваем мок метода query
        self.manager.query = MagicMock(return_value=[mock_record1, mock_record2])

        # 🎯 Вызываем тестируемый метод
        result = self.manager.get_entity_relationships()

        # ✅ Проверки (ассерты)
        self.assertIsInstance(result, pd.DataFrame, "Результат должен быть DataFrame")
        self.assertEqual(len(result), 2, "Должно быть 2 строки")

        # Проверка колонок
        expected_columns = [
            "source", "target", "weight", "description",
            "text_unit_ids", "updated_at", "created_at"
        ]
        self.assertListEqual(list(result.columns), expected_columns)

        # Проверка конкретных значений
        self.assertEqual(result.iloc[0]["source"], "Alice|PERSON")
        self.assertEqual(result.iloc[0]["target"], "Bob|PERSON")
        self.assertEqual(result.iloc[0]["weight"], 2.5)
        self.assertTrue(pd.isna(result.iloc[1]["weight"])) # None обрабатывается корректно

        # Проверка типа данных для weight
        self.assertEqual(result["weight"].dtype, "float64")

    def test02_get_entity_relationships_empty_result(self):
        """Тест: метод возвращает пустой DataFrame с правильной схемой при отсутствии связей."""
        # Мокаем пустой результат
        self.manager.query = MagicMock(return_value=[])

        result = self.manager.get_entity_relationships()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty, "DataFrame должен быть пустым")

        # Проверяем, что все ожидаемые колонки присутствуют
        expected_columns = [
            "source", "target", "weight", "description",
            "text_unit_ids", "updated_at", "created_at"
        ]
        self.assertListEqual(list(result.columns), expected_columns)

        # Проверяем типы данных пустых колонок
        self.assertEqual(result["weight"].dtype, "float64")
        self.assertEqual(result["source"].dtype, "string")



    def test03_get_entity_relationships_exception_handling(self):
        """Тест: при ошибке запроса метод возвращает пустой DataFrame с схемой, а не падает."""
        # Мокаем выброс исключения
        self.manager.query = MagicMock(side_effect=Exception("Neo4j connection failed"))

        result = self.manager.get_entity_relationships()

        # Метод не должен падать, а возвращать пустой DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

        expected_columns = [
            "source", "target", "weight", "description",
            "text_unit_ids", "updated_at", "created_at"
        ]
        self.assertListEqual(list(result.columns), expected_columns)

    def test04_connection_to_real_neo4g(self):

        load_dotenv()

        self.config = ManagerConfig(
            uri='neo4j://' + os.environ['URL'],
            user=os.environ['USER_NEO4J'],
            password=os.environ['PASSWORD'],
            name_db=os.environ['NAME_DB']
        )
        self.manager = Manager(self.config)
        res = self.manager.status()
        if 'error' in res.keys():
            logger.error("Neo4j status error: %s", res)
        self.assertNotIn("error", res.keys())
        result = self.manager.get_entity_relationships()
        logger.info("get_entity_relationships result:\n%s", result)
        self.assertGreater(len(result), 0, "Должно быть больше 0")


class TestGetEntitiesAndCommunitiesIntegration(unittest.TestCase):
    """Интеграционные тесты get_entities и _get_community с реальной Neo4j."""

    def setUp(self):
        load_dotenv()
        self.config = ManagerConfig(
            uri='neo4j://' + os.environ['URL'],
            user=os.environ['USER_NEO4J'],
            password=os.environ['PASSWORD'],
            name_db=os.environ['NAME_DB'],
        )
        self.manager = Manager(self.config)

    def tearDown(self):
        self.manager.close()

    def _assert_db_connected(self):
        res = self.manager.status()
        if 'error' in res:
            logger.error("Neo4j status error: %s", res)
        self.assertNotIn('error', res)

    def test01_get_entities(self):
        """get_entities возвращает непустой DataFrame с ожидаемой схемой."""
        self._assert_db_connected()

        result = self.manager.get_entities()
        logger.info("get_entities result:\n%s", result)

        expected_columns = [
            'id', 'title', 'type', 'description', 'data', 'updated_at', 'created_at',
        ]
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), expected_columns)
        self.assertGreater(len(result), 0, 'В базе должны быть сущности Entity')

        db_count = self.manager.query('MATCH (e:Entity) RETURN count(e) AS cnt')[0]['cnt']
        self.assertEqual(len(result), db_count, 'Число строк должно совпадать с count(e) в Neo4j')

        self.assertTrue(result['title'].notna().all(), 'title не должен быть пустым')
        self.assertTrue(result['type'].notna().all(), 'type не должен быть пустым')

    def test02_get_community(self):
        """_get_community возвращает непустой DataFrame с ожидаемой схемой."""
        self._assert_db_connected()

        result = self.manager.get_community()
        logger.info("get_community result:\n%s", result)

        expected_columns = ['id', 'title', 'level', 'parent', 'size', 'period']
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), expected_columns)
        self.assertGreater(len(result), 0, 'В базе должны быть узлы Community')

        db_count = self.manager.query('MATCH (c:Community) RETURN count(c) AS cnt')[0]['cnt']
        self.assertEqual(len(result), db_count, 'Число строк должно совпадать с count(c) в Neo4j')

        self.assertTrue(result['level'].notna().all(), 'level не должен быть пустым')
        self.assertTrue(result['size'].notna().all(), 'size не должен быть пустым')


if __name__ == '__main__':
    # Запуск тестов с подробным выводом
    unittest.main(verbosity=2)