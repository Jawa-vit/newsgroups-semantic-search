FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-download the embedding model so the container starts cold-start-free.
# The model (~90 MB) is baked into the image — this is intentional for
# production deployments where pulling from HuggingFace Hub at runtime
# would add latency and create a network dependency.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the service
# Note: --workers 1 because SemanticCache is in-process and not shared across
# worker processes. For multi-worker setups, migrate the cache to a shared
# store (e.g., a fast in-memory DB or a dedicated cache service).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
