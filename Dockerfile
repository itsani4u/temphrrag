# Simple Dockerfile for deploying the HR RAG Streamlit app to Google Cloud Run
FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss-cpu / pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects the PORT env var (defaults to 8080). Streamlit must
# bind to 0.0.0.0 on that port for Cloud Run health checks to pass.
ENV PORT=8080
EXPOSE 8080

# Use app_with_memory.py by default; swap to app.py if you want the
# no-memory version. You can also override this at deploy time.
CMD streamlit run app_with_memory.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
