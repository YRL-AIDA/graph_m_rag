FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
# Установка Python 3.11 из PPA deadsnakes и системных пакетов
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
        python3.11 python3.11-venv python3.11-dev \
        python3-opencv libglib2.0-0 wget git && \
    rm -rf /var/lib/apt/lists/*

# Установка pip для Python 3.11
RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --upgrade pip setuptools wheel

# Делаем Python 3.11 версией по умолчанию
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

COPY requirements.txt .
COPY pre_download.py .

# Устанавливаем PyTorch с поддержкой CUDA 12.4 (совместим с 12.8)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# Остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install doclayout_yolo ultralytics ftfy shapely pyclipper omegaconf

ENV MODELSCOPE_CACHE=/app/models
RUN python pre_download.py

COPY manager.py api.py cache_models.py ./

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]