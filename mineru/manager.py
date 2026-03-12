#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import tempfile
import json
import base64
from dataclasses import dataclass
from functools import lru_cache

import torch
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.enum_class import MakeMode

from modelscope import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient
from PIL import Image
import io


@dataclass
class ProcessingConfig:
    backend: str = "pipeline"
    method: str = "auto"
    lang: str = "en"
    formula_enable: bool = True
    table_enable: bool = True
    start_page_id: int = 0
    end_page_id: Optional[int] = None
    server_url: Optional[str] = None


class MinerUManager:
    _instance = None
    _vlm_client = None
    _pipeline_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MinerUManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._init_models()

    def _init_models(self):
        print("Инициализация MinerU моделей...")

        cache_dir = os.getenv("MODELSCOPE_CACHE", "/app/models")
        os.environ['MODELSCOPE_CACHE'] = cache_dir

        print(f"Используется кеш моделей: {cache_dir}")

        pdf_model_path = Path(cache_dir) / "OpenDataLab" / "PDF-Extract-Kit-1.0"
        vlm_model_path = Path(cache_dir) / "OpenDataLab" / "MinerU2.5-2509-1.2B"

        if not pdf_model_path.exists():
            print(f"Внимание: PDF модель не найдена в кеше: {pdf_model_path}")

        if not vlm_model_path.exists():
            print(f"Внимание: VLM модель не найдена в кеше: {vlm_model_path}")

        print("Модели готовы к использованию")

    @lru_cache(maxsize=1)
    def get_vlm_client(self):
        if self._vlm_client is None:
            print("Создание VLM клиента...")
            try:
                cache_dir = os.getenv("MODELSCOPE_CACHE", "/app/models")
                path_model = Path(cache_dir) / "OpenDataLab" / "MinerU2.5-2509-1.2B"
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    path_model,
                    dtype="auto",
                    device_map="auto"
                )
                processor = AutoProcessor.from_pretrained(
                    path_model,
                    use_fast=True
                )
                self._vlm_client = MinerUClient(
                    backend="transformers",
                    model=model,
                    processor=processor
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print("VLM клиент создан успешно на " + device)
            except Exception as e:
                print(f"Ошибка создания VLM клиента: {e}")
                raise
        return self._vlm_client

    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            client = self.get_vlm_client()

            image = Image.open(io.BytesIO(image_bytes))

            extracted_blocks = client.two_step_extract(image)

            # Convert the original image to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format=image.format if image.format else 'PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            return {
                "success": True,
                "results": {
                    "image_info": extracted_blocks,
                    "image_base64": img_base64
                },
                "format": "vlm"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "format": "vlm"
            }

    def process_pdf(
            self,
            pdf_bytes: bytes,
            config: ProcessingConfig,
            output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            if output_dir is None:
                temp_dir = tempfile.mkdtemp(prefix="mineru_")
                output_dir = temp_dir

            pdf_file_names = ["document"]
            pdf_bytes_list = [pdf_bytes]
            p_lang_list = [config.lang]

            if config.backend == "pipeline":
                pdf_bytes_list[0] = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, config.start_page_id, config.end_page_id
                )

                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                    pdf_bytes_list,
                    p_lang_list,
                    parse_method=config.method,
                    formula_enable=config.formula_enable,
                    table_enable=config.table_enable
                )

                idx = 0
                model_list = infer_results[idx]
                pdf_file_name = pdf_file_names[idx]
                local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, config.method)
                image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

                images_list = all_image_lists[idx]
                pdf_doc = all_pdf_docs[idx]
                _lang = lang_list[idx]
                _ocr_enable = ocr_enabled_list[idx]

                middle_json = pipeline_result_to_middle_json(
                    model_list, images_list, pdf_doc, image_writer,
                    _lang, _ocr_enable, config.formula_enable
                )

                image_dir = str(Path(local_image_dir).name)
                md_content = pipeline_union_make(
                    middle_json["pdf_info"],
                    MakeMode.MM_MD,
                    image_dir
                )

                content_list = pipeline_union_make(
                    middle_json["pdf_info"],
                    MakeMode.CONTENT_LIST,
                    image_dir
                )

                format_type = "pipeline"

            else:
                pdf_bytes_list[0] = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, config.start_page_id, config.end_page_id
                )

                pdf_file_name = pdf_file_names[0]
                local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, "vlm")
                image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

                middle_json, infer_result = vlm_doc_analyze(
                    pdf_bytes_list[0],
                    image_writer=image_writer,
                    backend=config.backend.replace("vlm-", ""),
                    server_url=config.server_url
                )

                image_dir = str(Path(local_image_dir).name)
                md_content = vlm_union_make(
                    middle_json["pdf_info"],
                    MakeMode.MM_MD,
                    image_dir
                )

                content_list = vlm_union_make(
                    middle_json["pdf_info"],
                    MakeMode.CONTENT_LIST,
                    image_dir
                )

                model_list = infer_result
                format_type = "vlm"

            # Convert images to base64
            images_base64 = {}
            images_dir_path = Path(local_image_dir)
            if images_dir_path.exists():
                for img_file in images_dir_path.glob("*"):
                    if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp',
                                                                          '.tiff', '.webp']:
                        with open(img_file, "rb") as f:
                            img_data = f.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            images_base64[img_file.name] = img_base64

            results = {
                "pdf_info": middle_json.get("pdf_info", {}),
                "middle_json": middle_json,
                "model_output": model_list,
                "markdown": md_content,
                "content_list": content_list,
                "output_dir": output_dir,
                "files": {
                    "middle_json": str(Path(local_md_dir) / "document_middle.json"),
                    "markdown": str(Path(local_md_dir) / "document.md"),
                    "content_list": str(Path(local_md_dir) / "document_content_list.json"),
                    "images_dir": local_image_dir,
                },
                "images_base64": images_base64
            }

            with open(results["files"]["middle_json"], "w") as f:
                json.dump(middle_json, f, ensure_ascii=False, indent=2)

            with open(results["files"]["markdown"], "w") as f:
                f.write(md_content)

            with open(results["files"]["content_list"], "w") as f:
                json.dump(content_list, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "format": format_type,
                "results": results
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "format": config.backend
            }