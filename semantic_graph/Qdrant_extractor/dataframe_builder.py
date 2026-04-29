import pandas as pd
from collections import defaultdict
from typing import Optional

def build_chunks_dataframe(adapter, doc_id_field="file_hash",doc_id: Optional[str] = None):
    rows = []

    # 🔥 Формируем фильтр Qdrant, если передан конкретный документ
    qdrant_filter = None
    if doc_id:
        qdrant_filter = {
            "must": [
                {"key": doc_id_field, "match": {"value": doc_id}}
            ]
        }

    for collection, point in adapter.iter_all_points(filter_payload=qdrant_filter):
        payload = point.get("payload", {})
        original = payload.get("original_element")
        if not isinstance(original, dict):
            continue

        # 🔥 2. Оставляем ТОЛЬКО элементы, где original_element.type == "text"
        if original.get("type") != "text":
            continue

        # 🔥 3. Берём текст из original_element.text, а не из верхнеуровневого "text"
        text = original.get("text")
        if not text:
            continue

        doc_id_val = payload.get(doc_id_field, "unknown_doc")

        rows.append({
            "document_id": doc_id_val,
            "chunk_id": f"{doc_id_val}_{point.get('id')}",
            "collection": collection,
            "text": text,  # 👈 Теперь текст из original_element
            "page": original.get("page_idx"),
            "element_index": payload.get("element_index"),
            "created_at": payload.get("created_at"),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["document_id", "page", "element_index"],
            na_position="last"
        ).reset_index(drop=True)

    return df