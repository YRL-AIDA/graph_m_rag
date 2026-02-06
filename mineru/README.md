# MinerU API Service

Docker-сервис для обработки PDF и изображений с использованием MinerU.

## Возможности

- 📄 Обработка PDF документов (pipeline и VLM бэкенды)
- 🖼️ Обработка изображений (PNG, JPG, TIFF, BMP)
- ⚡ Асинхронная обработка с отслеживанием статуса
- 📊 Извлечение текста, таблиц, формул и структуры документа
- 🔗 REST API с автоматической генерацией документации
- 🐳 Предзагруженные модели в Docker образе

## Быстрый старт

### 1. Сборка Docker образа

```bash
docker build -t mineru-api .
```

### 2. Запуск контейнера

```bash
docker run -p 8000:8000 --gpus all mineru-api
```

Или без GPU:

```bash
docker run -p 8000:8000 mineru-api
```

### 3. Использование API

Откройте документацию API: http://localhost:8000/docs

## Эндпоинты API

### POST /process
Загрузка и обработка документа.

**Параметры:**
- `file`: Файл для обработки (PDF или изображение)
- `backend`: Бэкенд обработки (`pipeline` или `vlm`)
- `method`: Метод обработки (`auto`, `txt`, `ocr`)
- `lang`: Язык документа (`ru`, `en`, `ch`, и др.)
- `formula_enable`: Обработка формул (true/false)
- `table_enable`: Обработка таблиц (true/false)
- `start_page`: Начальная страница (0-indexed)
- `end_page`: Конечная страница

**Ответ:**
```json
{
  "task_id": "uuid",
  "status": "processing",
  "message": "Документ принят в обработку",
  "download_links": {
    "status": "http://localhost:8000/status/{task_id}"
  }
}
```

### GET /status/{task_id}
Получение статуса задачи.

### GET /download/{task_id}/{file_path}
Скачивание файлов результата.

### DELETE /cleanup/{task_id}
Очистка ресурсов задачи.

## Параметры по умолчанию

### Для PDF:
```bash
backend=pipeline
method=auto
lang=ru
formula_enable=true
table_enable=true
```

### Для изображений:
Автоматически используется VLM бэкенд.

## Примеры использования

### cURL
```bash
# Обработка PDF
curl -X POST "http://localhost:8000/process" \
  -F "file=@document.pdf" \
  -F "backend=pipeline" \
  -F "lang=ru"

# Проверка статуса
curl "http://localhost:8000/status/{task_id}"

# Скачивание результатов
curl "http://localhost:8000/download/{task_id}/document.md" -o result.md
```

### Python
```python
import requests

# Загрузка файла
with open("test/document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process",
        files={"file": f},
        params={"backend": "pipeline", "lang": "ru"}
    )


print(('='*30)+"TASK" +('='*30) )    
print(response.json())
task_id = response.json()["task_id"]

# Проверка статуса
result = requests.get(f"http://localhost:8000/status/{task_id}").json()
print(('='*30)+"RESULT" +('='*30) ) 
print(result)    

# Скачивание результатов
md_content = requests.get(
    result['results']['download_links']['markdown']
).text

print(('='*30)+"Files" +('='*30) ) 
print(md_content)
```

## Структура результатов

После обработки доступны:
- `document.md` - Markdown версия документа
- `document_middle.json` - Промежуточное JSON представление
- `document_content_list.json` - Структурированный список контента
- `images/` - Директория с извлеченными изображениями

## Конфигурация Docker

### Переменные окружения:
- `MODELSCOPE_CACHE` - Путь к кешу моделей (по умолчанию: `/app/models`)
- `BASE_URL` - Базовый URL для ссылок скачивания

### Использование GPU:
```bash
docker run -p 8000:8000 --gpus all mineru-api
```

## Мониторинг и логи

- Логи доступны через `docker logs <container_id>`
- Метрики здоровья: `GET /health`
- Документация API: `GET /docs` или `GET /redoc`

## Устранение неполадок

1. **Модели не загружаются**
   - Проверьте доступ к интернету при сборке
   - Убедитесь, что на диске достаточно места (требуется ~10GB)

2. **Ошибки CUDA/GPU**
   - Проверьте наличие драйверов NVIDIA
   - Используйте `--gpus all` при запуске
   - Или запускайте без GPU: удалите `--gpus all`

3. **Недостаточно памяти**
   - Увеличьте лимиты памяти Docker
   - Используйте `backend=pipeline` вместо `vlm`

## Технические детали

### Предзагруженные модели:
1. **PDF-Extract-Kit-1.0** - для pipeline обработки PDF
2. **MinerU2.5-2509-1.2B** - для VLM обработки

### Используемые технологии:
- FastAPI для REST API
- MinerU для обработки документов
- ModelScope для управления моделями
- PyTorch для инференса

### Оптимизации:
- Модели загружаются один раз при запуске
- Кеширование результатов обработки
- Асинхронная обработка файлов
- Многостадийная сборка Docker образа
