# Document Intelligence API

[![CI](https://github.com/YOUR_USERNAME/doc-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/doc-intelligence/actions/workflows/ci.yml)

A fully local RAG (Retrieval Augmented Generation) system. Upload PDF or TXT documents, ask questions in natural language, and get answers grounded in your documents — with source references. No API keys required.

## Architecture

```
PDF/TXT → Chunking → Embeddings (MiniLM) → ChromaDB
                                                ↓
Question → Embeddings → Similarity Search → Top-K Chunks → Llama 3.2 → Answer
```

## Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.2 3B via Ollama (fully local) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Vector DB | ChromaDB |
| API | FastAPI |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

## Quickstart

### Prerequisites
- Docker Desktop installed and running
- Git

### 1. Clone and start
```bash
git clone https://github.com/YOUR_USERNAME/doc-intelligence.git
cd doc-intelligence
docker-compose up --build
```
First run takes 5–10 minutes (downloads Ollama image + builds API image).

### 2. Pull the LLM model (one-time, in a new terminal)
```bash
docker exec doc-intel-ollama ollama pull llama3.2
```

### 3. Use the API
API docs available at: http://localhost:8000/docs

**Upload a document:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

**List indexed documents:**
```bash
curl http://localhost:8000/sources
```

**Delete a document:**
```bash
curl -X DELETE http://localhost:8000/source/your_document.pdf
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| POST | `/ingest` | Upload and index a PDF or TXT file |
| POST | `/query` | Ask a question, get answer + sources |
| GET | `/sources` | List all indexed documents |
| DELETE | `/source/{filename}` | Remove a document from the index |

## Running locally without Docker

```bash
# Install Ollama from ollama.com and pull the model
ollama pull llama3.2

# Create venv and install deps
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

## Running tests

```bash
pytest tests/ -v
```
