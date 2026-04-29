from config import QDRANT_URL, QDRANT_API_KEY, OUTPUT_DIR
from qdrant_adapter import QdrantStreamAdapter
from dataframe_builder import build_chunks_dataframe
from export_graphrag import export_to_graphrag


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