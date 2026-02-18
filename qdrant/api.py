from fastapi import FastAPI, Depends, HTTPException
from qdrant_client import QdrantClient, models
from contextlib import asynccontextmanager
from config.settings import settings
import os
import uvicorn


QDRANT_URL = f"{settings.host}:{settings.port}"
COLLECTION_NAME = "my_collection"

class QdrantClientSingleton:
    def __init__(self):
        if settings.api_key:
            self.client = QdrantClient(url=QDRANT_URL, api_key=settings.api_key)
        else:
            self.client = QdrantClient(url=QDRANT_URL)

qdrant_singleton = QdrantClientSingleton()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # During startup, ensure client is ready and collection exists (if needed)
    print("Starting up and checking Qdrant connection...")
    try:
        # Example: create collection if it doesn't exist
        if not qdrant_singleton.client.collection_exists(collection_name=COLLECTION_NAME):
            qdrant_singleton.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=4, distance=models.COSINE), # Example size
            )
            print(f"Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        print(f"Could not connect to Qdrant or create collection: {e}")

    yield
    qdrant_singleton.client.close()
    print("Shutting down Qdrant client.")

app = FastAPI(lifespan=lifespan)

def get_qdrant_client() -> QdrantClient:
    return qdrant_singleton.client

@app.get("/search")
async def search_vectors(query_vector: str, client: QdrantClient = Depends(get_qdrant_client)):
    try:
        vector = [float(x) for x in query_vector.split(',')]
        if len(vector) != 4:
            raise HTTPException(status_code=400, detail="Vector size must be 4.")

        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=5, # Return top 5 results
        )
        return {"results": [{"id": res.id, "score": res.score, "payload": res.payload} for res in search_results]}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid vector format. Use comma-separated floats.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upsert_points")
async def upsert_points(points: list[dict], client: QdrantClient = Depends(get_qdrant_client)):
    try:
        points_structs = [
            models.PointStruct(
                id=p['id'],
                vector=p['vector'],
                payload=p.get('payload', {})
            ) for p in points
        ]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_structs
        )
        return {"status": "upserted", "count": len(points)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_api(
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        log_level: str = "info") -> None:
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )