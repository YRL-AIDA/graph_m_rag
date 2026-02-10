# Graph-based RAG System

This repository contains a RAG (Retrieval Augmented Generation) system with graph-based retrieval capabilities.

## Project Structure

```
graph_m_rag/
├── src/
│   └── graph_m_rag/
│       ├── __init__.py
│       ├── mineru/
│       │   ├── __init__.py
│       │   ├── api.py
│       │   ├── mineru_wrapper.py
│       │   ├── cache_models.py
│       │   └── pre_download.py
│       └── qdrant/
│           ├── __init__.py
│           └── api.py
├── pyproject.toml
└── README.md
```

## Features

- Document processing with MinerU
- Vector storage and retrieval with Qdrant
- RESTful API endpoints
- Support for PDF and image processing

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Start the services
cd src/graph_m_rag/mineru
python api.py

# Or use the docker-compose setup
docker-compose up
```
