#!/usr/bin/env python3


import os
from modelscope import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():

    cache_dir = os.getenv("MODELSCOPE_CACHE", "/app/models")
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"Начинаем загрузку моделей в {cache_dir}")
    
    try:
        logger.info("Загружаем PDF-Extract-Kit-1.0...")
        pdf_model_path = snapshot_download(
            "OpenDataLab/PDF-Extract-Kit-1.0",
            cache_dir=cache_dir
        )
        logger.info(f"PDF модель загружена: {pdf_model_path}")
    except Exception as e:
        logger.error(f"Ошибка загрузки PDF модели: {e}")
    
    try:
        logger.info("Загружаем MinerU2.5-2509-1.2B...")
        vlm_model_path = snapshot_download(
            "OpenDataLab/MinerU2.5-2509-1.2B",
            cache_dir=cache_dir
        )
        logger.info(f"VLM модель загружена: {vlm_model_path}")
    except Exception as e:
        logger.error(f"Ошибка загрузки VLM модели: {e}")
    
    logger.info("Загрузка моделей завершена")

if __name__ == "__main__":
    download_models()