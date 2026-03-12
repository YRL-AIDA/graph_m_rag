# graph_m_rag

# Инструкция по запуску сервиса Graph M-RAG
## Требования
- **Python 3.11**
- **Docker** и **Docker Compose**
- **GPU** (для работы MinerU с CUDA)
  - Рекомендуемая версия CUDA: **12.0+**
  - Поддерживаемые драйверы NVIDIA: **535.104.05+**
  - Минимальный объем видеопамяти: **8 GB**
  - Рекомендуемый объем видеопамяти: **16 GB+**

### Установка NVIDIA Container Toolkit
Для работы контейнеров с GPU необходимо установить NVIDIA Container Toolkit:

#### Для Ubuntu/Debian:
```bash
# Добавление репозитория NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Обновление списка пакетов и установка
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Перезапуск Docker
sudo systemctl restart docker
```

## Запуск через Docker Compose (Рекомендуемый)
### Шаг 1: Подготовка переменных окружения
Создайте файл `.env` в корневой директории проекта:
```bash
cp .env.example .env
```
Или создайте вручную со следующим содержимым:
```env
# MinIO credentials
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=minio123
MINIO_BUCKET=documents
# Qdrant API Key (опционально)
QDRANT_API_KEY=
```
### Шаг 2: Запуск всех сервисов
```bash
docker-compose up -d --build
```
Флаги:
- `-d` — запуск в фоновом режиме
- `--build` — пересобрать образы
### Шаг 3: Проверка статуса
```bash
docker-compose ps
```
Все сервисы должны быть в статусе `Up`.
### Шаг 4: Просмотр логов
```bash
# Логи всех сервисов
docker-compose logs -f
# Логи конкретного сервиса
docker-compose logs -f mineru
docker-compose logs -f minio
docker-compose logs -f qdrant
```
### Шаг 5: Остановка сервисов
```bash
# Остановить без удаления данных
docker-compose down
# Остановить и удалить volumes (данные будут потеряны!)
docker-compose down -v
```
### Шаг 6: Запуск основного приложения
```bash
cd app
# Установка зависимостей
pip install -r ../requirements.txt
# Настройка переменных окружения
export S3_URL=http://localhost:9000
export S3_ACCESS_KEY=minio
export S3_SECRET_KEY=minio123
# Запуск
python src/main.py
```
