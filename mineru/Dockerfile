FROM python:3.11-slim

RUN apt-get update && apt-get install -y python3-opencv libglib2.0-0 wget git

WORKDIR /app

COPY requirements.txt .
COPY pre_download.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install doclayout_yolo ultralytics ftfy shapely pyclipper omegaconf
ENV MODELSCOPE_CACHE=/app/models
RUN python pre_download.py

COPY mineru_wrapper.py api.py cache_models.py ./

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]