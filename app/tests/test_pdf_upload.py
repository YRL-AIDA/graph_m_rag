import os

import requests


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
        data = {
            "backend": "pipeline",
            "method": "auto",
            "lang": "ru",
            "formula_enable": "true",
            "table_enable": "true"
        }

        response = requests.post(url, files=files, data=data)

    # Проверяем статус ответа
    assert response.status_code == 200

    # Проверяем структуру ответа
    response_data = response.json()
    assert "task_id" in response_data
    assert "status" in response_data
    assert "message" in response_data
    assert response_data["status"] == "processing"
    assert response_data["message"] == "Документ принят в обработку"

    print("Тест загрузки PDF успешно пройден!")
    print(f"Task ID: {response_data['task_id']}")


if __name__ == "__main__":
    # Убедимся, что сервер запущен перед выполнением тестов
    # В реальной ситуации вы можете запустить сервер в фоновом режиме
    try:
        # Проверяем доступность сервера
        health_response = requests.get("http://localhost:9191/health")
        if health_response.status_code == 200:
            print("Сервер доступен, запускаем тесты...")
            test_upload_pdf_endpoint()
        else:
            print("Сервер не отвечает. Убедитесь, что API запущено на http://localhost:9191")
    except requests.exceptions.ConnectionError:
        print("Не удается подключиться к серверу. Убедитесь, что API запущено на http://localhost:9191")