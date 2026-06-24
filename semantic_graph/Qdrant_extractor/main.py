import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, QDRANT_API_KEY, QDRANT_URL
from dataframe_builder import build_chunks_dataframe
from export_graphrag import export_to_graphrag
from qdrant_adapter import QdrantStreamAdapter


def main():
    adapter = QdrantStreamAdapter(
        base_url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    print("Loading data from Qdrant...")
    df = build_chunks_dataframe(adapter)

    print(df.head())
    print(f"Total chunks: {len(df)}")

    export_to_graphrag(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
