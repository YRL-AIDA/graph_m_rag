import os
import pandas as pd


def export_to_graphrag(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------
    # documents.csv
    # -------------------
    documents = (
        df[["document_id"]]
        .drop_duplicates()
        .rename(columns={"document_id": "id"})
    )

    documents["title"] = documents["id"]

    documents.to_csv(f"{output_dir}/documents.csv", index=False)

    # -------------------
    # text_units.csv
    # -------------------
    text_units = df.copy()

    text_units = text_units.rename(columns={
        "chunk_id": "id",
        "document_id": "document_id",
        "text": "text"
    })

    text_units = text_units[[
        "id",
        "document_id",
        "text",
        "page",
        "element_index"
    ]]

    text_units.to_csv(f"{output_dir}/text_units.csv", index=False)

    print(f"Saved to {output_dir}")