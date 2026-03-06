import os
from pathlib import Path

import requests

def test_upload_pdf_directory():
    """
    Тестирование загрузки всех PDF файлов из директории через API
    """
    # Директория с PDF файлами для тестирования
    pdf_directory = "/home/sunveil/Documents/projects/laba/graph-m-rag/data/MMLongBench-Doc/data/documents"

    # Проверяем существование директории
    if not os.path.exists(pdf_directory):
        print(f"Директория {pdf_directory} не найдена")
        return

    # Находим все PDF файлы в директории
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))

    if not pdf_files:
        print(f"В директории {pdf_directory} не найдено PDF файлов")
        return

    # Отправляем POST запрос с каждым PDF файлом на endpoint /upload-pdf
    url = "http://localhost:9191/upload-pdf"

    for pdf_file_path in pdf_files:
        print(f"Загружаем файл: {pdf_file_path.name}")

        with open(pdf_file_path, "rb") as pdf_file:
            files = {"file": (pdf_file_path.name, pdf_file, "application/pdf")}

            response = requests.post(url, files=files)

        # Проверяем статус ответа
        assert response.status_code == 200

        # Проверяем структуру ответа
        response_data = response.json()
        assert "status" in response_data
        assert "message" in response_data
        assert "file_hash" in response_data
        assert "s3_path" in response_data
        assert "mineru_result_path" in response_data
        assert "embeddings_computed" in response_data
        assert "processing_time" in response_data
        assert response_data["status"] == "success" or response_data["status"] == "already_processed"

        print(f"Файл {pdf_file_path.name} успешно загружен!")
        print(f"Response: {response_data}")

def test_upload_pdf_endpoint():
    """
    Тестирование загрузки PDF файла через API
    """
    # Проверяем наличие тестового PDF файла
    test_pdf_path = "app/tests/data/prospectus_en.pdf"

    # Отправляем POST запрос с PDF файлом на endpoint /upload-pdf
    url = "http://localhost:9191/upload-pdf"

    with open(test_pdf_path, "rb") as pdf_file:
        files = {"file": ("sample.pdf", pdf_file, "application/pdf")}

        response = requests.post(url, files=files)

    # Проверяем статус ответа
    assert response.status_code == 200

    # Проверяем структуру ответа
    response_data = response.json()
    assert "status" in response_data
    assert "message" in response_data
    assert "file_hash" in response_data
    assert "s3_path" in response_data
    assert "mineru_result_path" in response_data
    assert "embeddings_computed" in response_data
    assert "processing_time" in response_data
    assert response_data["status"] == "success"
    assert "processed with MinerU" in response_data["message"]

    print("Тест загрузки PDF успешно пройден!")
    print(f"Response: {response_data}")


if __name__ == "__main__":
    # Убедимся, что сервер запущен перед выполнением тестов
    # В реальной ситуации вы можете запустить сервер в фоновом режиме
    try:
        # Проверяем доступность сервера
        health_response = requests.get("http://localhost:9191/health")
        if health_response.status_code == 200:
            print("Сервер доступен, запускаем тесты...")
            test_upload_pdf_directory()
        else:
            print("Сервер не отвечает. Убедитесь, что API запущено на http://localhost:9191")
    except requests.exceptions.ConnectionError:
        print("Не удается подключиться к серверу. Убедитесь, что API запущено на http://localhost:9191")