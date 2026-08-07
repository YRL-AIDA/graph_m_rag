FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY semantic_graph/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK wordnet for lemmatizer (B5: _normalize_entity_title)
RUN python -m nltk.downloader -d /usr/local/share/nltk_data wordnet punkt punkt_tab

# Copy source code: semantic_graph + app (needed for EmbeddingClient)
COPY semantic_graph/ ./semantic_graph/
COPY app/ ./app/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 9595

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:9595/docs || exit 1

CMD ["python", "/app/semantic_graph/semantic_index.py"]
