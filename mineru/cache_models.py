#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def check_model_cache():
    cache_dir = Path(os.getenv("MODELSCOPE_CACHE", "/app/models"))
    
    pdf_model = cache_dir / "OpenDataLab" / "PDF-Extract-Kit-1.0"
    vlm_model = cache_dir / "OpenDataLab" / "MinerU2.5-2509-1.2B"
    
    if not pdf_model.exists():
        print(f"PDF модель не найдена: {pdf_model}")
        sys.exit(1)
    
    if not vlm_model.exists():
        print(f"VLM модель не найдена: {vlm_model}")
        sys.exit(1)
    
    print(f"✓ PDF модель: {pdf_model}")
    print(f"✓ VLM модель: {vlm_model}")
    print("Кеш моделей проверен успешно")
    
    return True

if __name__ == "__main__":
    check_model_cache()