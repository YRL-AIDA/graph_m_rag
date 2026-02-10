from app.src.services.mineru_service import mineru_service

# Анализ PDF файла
result = mineru_service.analyze_pdf(
    pdf_data="test/document2.pdf",
    filename="document2.pdf",
    page_limit=10,
    use_cache=True
)

if result.get("status") == "success":
    # Извлечение текста
    text = mineru_service.extract_text_from_result(result)

    # Извлечение таблиц
    tables = mineru_service.extract_tables_from_result(result)

    # Извлечение изображений
    images = mineru_service.extract_images_from_result(result)

    print(f"Извлечено: {len(text)} символов текста, {len(tables)} таблиц, {len(images)} изображений")
else:
    print(f"Ошибка анализа: {result.get('error')}")
