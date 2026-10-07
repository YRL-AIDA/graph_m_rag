# Graph-M-RAG

**Multi-Graph Retrieval-Augmented Generation for document question answering.**

Graph-M-RAG is a production-grade RAG system that combines **structural document graphs** (layout hierarchy, reading order) with **semantic knowledge graphs** (entities, relationships, communities) to answer complex questions over PDF documents. It uses a multimodal LLM (Qwen3-VL-32B) to process text, tables, and images jointly.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Graph-M-RAG System                          │
├──────────────────────┬──────────────────────┬───────────────────────┤
│   Document Pipeline  │   Retrieval & Graph  │   Demo Application    │
│                      │                      │                       │
│  PDF ─→ MinerU ─→   │  Qdrant (vectors) ──→│  Web UI (4 tabs)     │
│  Regions + Images    │  Neo4j (graphs) ────→│  Strategy comparison │
│       ↓              │  ┌────────────────┐  │  Graph visualization │
│  MinIO (S3) storage  │  │ Structural     │  │  Evidence highlight  │
│       ↓              │  │ Graph (ORDER,  │  │                       │
│  Embedding service ─→│  │  PARENT, SECT) │  │                       │
│  Qdrant (vectors)    │  ├────────────────┤  │                       │
│       ↓              │  │ Semantic Graph │  │                       │
│  Neo4j (structural   │  │ (Entity,       │  │                       │
│  graph: regions,     │  │  Community,    │  │                       │
│  hierarchy, order)   │  │  RELATED)      │  │                       │
│       ↓              │  └────────────────┘  │                       │
│  Semantic Graph ────→│  BFS Crawler ────→   │                       │
│  (Entity extraction, │  Context Assembly    │                       │
│   Leiden clustering, │       ↓              │                       │
│   Community reports) │  LLM (Qwen3-VL-32B)  │                       │
│                      │  ─→ Answer           │                       │
└──────────────────────┴──────────────────────┴───────────────────────┘
```

### Core Components

| Component | Role | Technology |
|-----------|------|------------|
| **MinerU** | PDF → structured regions (text, tables, images, headings) | MinerU pipeline |
| **MinIO** | S3-compatible object storage for PDFs, images, results | MinIO |
| **Qdrant** | Vector database for embedding search (text blocks, entities, communities) | Qdrant |
| **Neo4j** | Graph database for structural layout graph + semantic knowledge graph | Neo4j 5.x |
| **Embedding Service** | Text + image embedding generation | Qwen3-Embedding |
| **Reranker** | Cross-encoder reranking of retrieved blocks | Qwen3-VL-Reranker |
| **LLM** | Multimodal answer generation (text + images + tables) | Qwen3-VL-32B-Thinking |
| **Semantic Graph** | Entity extraction, Leiden clustering, community reports | LLM + graspologic |
| **Demo App** | Interactive web UI for demonstrations | FastAPI + JS |

---

## Key Features

- **Dual-graph retrieval**: structural graph (document layout, reading order) + semantic graph (entities, communities, relationships)
- **Multimodal understanding**: processes text, tables, and images in a single LLM call
- **Hierarchical document regions**: PDF → regions (text blocks, headings, tables, images, footnotes) with layout-aware indexing
- **Entity extraction & clustering**: LLM-based entity/relationship extraction → Leiden clustering → community reports
- **Cross-graph bridges**: connects structural regions to semantic entities for richer context
- **BFS graph crawler**: unified traversal across both graphs for comprehensive evidence gathering
- **Multi-strategy comparison**: baseline RAG, semantic-only, structural-only, full Graph-M-RAG
- **MMR diversity reranking**: Maximal Marginal Relevance to reduce context redundancy
- **Iterative retrieval**: feedback-driven multi-round search for complex questions
- **Question decomposition**: LLM-based breakdown of multi-hop questions into sub-questions
- **Evidence gate**: anti-over-abstention heuristics to reduce false "Not answerable" responses
- **Context budget**: configurable character limits per source type to fit LLM context windows
- **Image captioning**: VLM-generated descriptions for caption-less images/tables
- **Interactive demo**: web UI with pipeline visualization, graph exploration, and strategy comparison

---

## System Requirements

### Hardware

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA with CUDA 12.0+, driver ≥535.104.05 |
| **VRAM** | 16 GB+ recommended (8 GB minimum) |
| **RAM** | 32 GB+ |
| **Storage** | 50 GB+ for models and data |

### Software

- **Python** 3.11+
- **Docker** & **Docker Compose**
- **NVIDIA Container Toolkit** (for GPU passthrough)

### External Services

These services run outside Docker and must be deployed separately:

| Service | Purpose | Reference |
|---------|---------|-----------|
| **Qwen3-Embedding** | Text/image embedding generation | [qwen-embedding-service](https://github.com/YRL-AIDA/qwen-embedding-service) |
| **Qwen3-VL-Reranker** | Cross-encoder reranking | Same service, `/reranker` endpoint |
| **Qwen3-VL-32B** | Multimodal answer generation | VLLM deployment |

---

## Quick Start (Docker Compose)

### 1. Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin_password
MINIO_ACCESS_KEY=minio
MINIO_SECRET_KEY=minio123
MINIO_BUCKET=pdf-processing

# Qdrant
QDRANT_API_KEY=

# Embedding service (external)
EMBEDDING_BASE_URL=http://192.168.19.127:10115/embedding

# S3/MinIO client (for the application)
S3_URL=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio123
S3_VERIFY_TLS=false
S3_BUCKET_NAME=pdf-processing
```

### 3. Launch Infrastructure Services

```bash
docker-compose up -d --build
```

This starts:
| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| **MinIO** | `rag-minio` | 9000 (API), 9001 (Console) | S3 object storage |
| **MinerU** | `rag-mineru` | 8001 | PDF → structured regions |
| **Qdrant** | `rag-qdrant` | 16333 | Vector database |
| **Neo4j** | `rag-neo4j` | 7474 (Browser), 7687 (Bolt) | Graph database |
| **Semantic Index** | `rag-semantic-index` | 9595 | Entity extraction, clustering, reports |

### 4. Start the Main Application

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
export S3_URL=http://localhost:9000
export S3_ACCESS_KEY=minio
export S3_SECRET_KEY=minio123

# Run the API server
cd app
python src/main.py
```

The API is now available at `http://localhost:9191`.

### 5. Start the Demo Application (Optional)

```bash
python demo/backend/demo_server.py --port 8282
```

Open `http://localhost:8282` in your browser.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_URL` | `http://localhost:9000` | MinIO S3 endpoint |
| `S3_ACCESS_KEY` | `minio` | S3 access key |
| `S3_SECRET_KEY` | `minio123` | S3 secret key |
| `S3_BUCKET_NAME` | `pdf-processing` | Default S3 bucket |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `16333` | Qdrant HTTP port |
| `QDRANT_API_KEY` | — | Qdrant API key |
| `MINERU_HOST` | `http://localhost` | MinerU service host |
| `MINERU_PORT` | `8001` | MinerU service port |
| `EMBEDDING_BASE_URL` | `http://192.168.19.127:10115/embedding` | Embedding service URL |
| `RERANKER_BASE_URL` | `http://192.168.19.127:10115/reranker` | Reranker service URL |
| `LLM_BASE_URL` | `http://192.168.19.127:8888/v1` | LLM API URL (OpenAI-compatible) |
| `LLM_MODEL_NAME` | `Qwen/Qwen3-VL-32B-Thinking` | LLM model name |
| `LLM_MAX_TOKENS` | `8192` | Max output tokens |
| `MAX_CONTEXT_CHARS` | `60000` | Total context character budget |
| `MAX_IMAGES` | `8` | Max images/tables in context |

### Context Budget

The context budget controls how much text from each source reaches the LLM:

| Source | Fraction of `MAX_CONTEXT_CHARS` | Purpose |
|--------|-------------------------------|---------|
| Qdrant (primary) | by block count (`limit=30`) | Main retrieval |
| Entities | 0.08 | Semantic entities matched to regions |
| Communities | 0.10 | Community reports |
| Cross-graph | 0.05 | Bridge connections between graphs |
| ORDER neighbors | 0.04 | Reading-order neighbors |
| BFS crawler | 0.10 | Unified graph traversal |
| Embedding search | 0.05 | Entity/community embedding search |

---

## API Endpoints

### Document Processing

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-pdf` | Upload and process a PDF document |
| `GET` | `/uploaded-files` | List uploaded documents |
| `GET` | `/collections` | List Qdrant collections |
| `POST` | `/collections` | Create a Qdrant collection |
| `DELETE` | `/collections/{name}` | Delete a Qdrant collection |

### Question Answering

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask-document` | Ask a question with full configuration |
| `GET` | `/ask-document` | Web interface for asking questions |
| `POST` | `/demonstration` | Demo endpoint with strategy selection |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check for all services |
| `GET` | `/` | API root with endpoint list |

### Semantic Graph (port 9595)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/process-document` | Extract entities/relationships from document chunks |
| `GET` | `/clastrize_graph` | Leiden clustering of the entity graph |
| `GET` | `/create_community_report` | Generate LLM reports for communities |
| `GET` | `/compute_entity_embeddings` | Compute embeddings for entities |
| `GET` | `/compute_community_embeddings` | Compute embeddings for communities |

### Question Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_hash` | string | — | Document identifier |
| `question` | string | — | Question text |
| `limit` | int | `30` | Number of blocks to retrieve |
| `use_llm` | bool | `false` | Generate LLM answer |
| `use_reranker` | bool | `false` | Enable cross-encoder reranking |
| `use_mmr_reranker` | bool | `false` | Enable MMR diversity reranking |
| `use_semantic_graph` | bool | `false` | Enrich with semantic graph |
| `use_structured_graph` | bool | `false` | Enrich with structural graph |
| `use_bfs_crawler` | bool | auto | Unified BFS graph traversal |
| `use_iterative_search` | bool | `false` | Multi-round iterative retrieval |
| `use_question_decomposition` | bool | `false` | Decompose complex questions |
| `answer_format` | string | — | Expected format: `Int`, `Float`, `List`, `Str` |

---

## Demo Application

The demo provides an interactive web UI for showcasing Graph-M-RAG capabilities:

```
demo/
├── README.md                 # Demo documentation
├── sample_questions.json     # Pre-built questions by document type
├── backend/
│   └── demo_server.py        # FastAPI proxy + strategy comparison
└── frontend/
    ├── index.html            # Dashboard (4 tabs)
    ├── styles.css
    └── app.js
```

### Demo Strategies

| Strategy | Graphs Used | Description |
|----------|-------------|-------------|
| `baseline` | None | Classic RAG: top-30 Qdrant blocks → LLM |
| `semantic` | Semantic | + entities and communities |
| `structural` | Structural | + reading-order neighbors and parent elements |
| `both` | Both | Full Graph-M-RAG with BFS traversal |

### Offline Mode

```bash
python demo/backend/demo_server.py --port 8282 --mock
```

Runs the demo with canned data — no infrastructure required.

---

## Project Structure

```
graph_m_rag/
├── app/                          # Main application
│   ├── src/
│   │   ├── api.py                # FastAPI server (port 9191)
│   │   ├── main.py               # Entry point
│   │   ├── llm_client.py         # OpenAI-compatible LLM client
│   │   ├── qdrant_client_api.py  # Qdrant vector DB client
│   │   ├── minio_client.py       # MinIO S3 client
│   │   ├── mineru_client.py      # MinerU PDF processing client
│   │   ├── qwen3_emb_client.py   # Embedding service client
│   │   ├── reranker_client.py    # Cross-encoder reranker client
│   │   ├── image_captioner.py    # VLM caption generation
│   │   ├── iterative_search.py   # Multi-round iterative retrieval
│   │   ├── question_decomposer.py # Multi-hop question decomposition
│   │   ├── schemas/              # Pydantic request/response models
│   │   └── utils/                # Answer formatting, MMR reranking, data models
│   ├── config/
│   │   └── settings.py           # Centralized configuration
│   ├── static/
│   │   └── ask-document.html     # Web UI for question answering
│   └── tests/                    # Evaluation and test scripts
├── semantic_graph/               # Semantic knowledge graph module
│   ├── semantic_index.py         # FastAPI server (port 9595)
│   ├── graphrag.py               # Entity/relationship extraction
│   ├── manager.py                # Neo4j CRUD operations
│   ├── clasterization.py         # Leiden clustering
│   ├── create_community_report.py # Community report generation
│   ├── config.py                 # Module configuration
│   ├── traversal.py              # Unified graph traversal
│   ├── embeddings.py             # Entity/community embeddings
│   ├── neo4j_service.py          # Neo4j document indexing
│   ├── dtype/                    # Pydantic models
│   ├── prompts/                  # LLM extraction prompts
│   └── Qdrant_extractor/         # Qdrant chunk extraction
├── documet_index/                # Structural document graph
│   ├── manager.py                # Neo4j graph construction from MinerU results
│   ├── neo4j_service.py          # Neo4j connection management
│   ├── region_classifier.py      # Region type classification
│   └── dtype/                    # Document and region models
├── mineru/                       # MinerU PDF processing service
│   ├── manager.py                # PDF → structured regions pipeline
│   ├── api.py                    # FastAPI server (port 8001)
│   └── Dockerfile
├── qdrant/                       # Qdrant vector DB service
│   └── api.py                    # FastAPI wrapper
├── demo/                         # Demo application
│   ├── backend/demo_server.py    # Demo proxy server (port 8282)
│   └── frontend/                 # Web UI (HTML, CSS, JS)
├── connect_graphs.py             # Cross-graph bridge script
├── docker-compose.yml            # Infrastructure orchestration
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment template
```

---

## Data Flow

### Document Ingestion

```
PDF Upload
  → MinIO (S3 storage)
  → MinerU (PDF → regions: text, tables, images, headings)
  → Embedding service (text + image embeddings)
  → Qdrant (vector index)
  → Neo4j (structural graph: regions, hierarchy, reading order)
  → Semantic Graph service (entity extraction → Leiden clustering → community reports)
  → Neo4j (semantic graph: entities, communities, relationships)
  → Qdrant (entity + community embeddings)
```

### Question Answering

```
Question
  → Qdrant search (top-k text blocks by embedding similarity)
  → Reranker (cross-encoder re-ranking)
  → MMR (diversity filtering)
  → Structural graph enrichment (ORDER neighbors, PARENT hierarchy)
  → Semantic graph enrichment (entities, communities, relationships)
  → BFS graph crawler (unified traversal)
  → Context assembly (budget-aware deduplication)
  → LLM (Qwen3-VL-32B: text + images + tables)
  → Answer
```

---

## Evaluation

The project includes comprehensive evaluation scripts in `app/tests/`:

| Script | Purpose |
|--------|---------|
| `evaluate_strategies.py` | Compare RAG strategies on benchmark datasets |
| `strategy_grid_test.py` | Grid search over strategy parameters |
| `structural_graph_test.py` | Test structural graph enrichment |
| `evaluate_structural_graph_test.py` | Evaluate structural graph impact |
| `consistency_test.py` | Test answer consistency across runs |
| `per_question_report.py` | Per-question detailed analysis |
| `mmlong_strategy_grid_test.py` | MMLongBench-Doc strategy grid |
| `mmlong_evaluate_strategies.py` | MMLongBench-Doc evaluation |

---

## License

See `LICENSE` file.

## Citation

If you use Graph-M-RAG in your research, please cite:

```bibtex
@software{graph-m-rag,
  title = {Graph-M-RAG: Multi-Graph Retrieval-Augmented Generation},
  year = {2025},
  url = {https://github.com/YRL-AIDA/graph-m-rag}
}
