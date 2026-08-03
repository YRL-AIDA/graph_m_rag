"""Конфигурация pytest для entity_extraction_test/tests/."""

import sys
from pathlib import Path


def pytest_configure(config):
    """Регистрация кастомных маркеров."""
    config.addinivalue_line("markers", "integration: тесты, требующие доступа к внешним данным (HF, API)")
    config.addinivalue_line("markers", "slow: медленные тесты (>1 сек)")


# Добавляем родительскую директорию в sys.path для импорта testdata
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
