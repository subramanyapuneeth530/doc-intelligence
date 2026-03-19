import logging
import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.ingest import delete_source, ingest_file, list_sources
from app.models import (
    DeleteResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourcesResponse,
)
from app.retriever import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Intelligence API",
    description="A fully local RAG system — upload documents, ask questions, get answers with sources.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Document Intelligence API is running."}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF or TXT file to index it into the vector store."""
    allowed = {"pdf", "txt"}
    ext = file.filename.lower().split(".")[-1]
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Only PDF and TXT files are supported. Got: .{ext}")

    # Save upload to a temp file so loaders can read it from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = ingest_file(tmp_path, file.filename)
        logger.info(f"Ingested '{file.filename}' → {chunks} chunks")
        return IngestResponse(
            message="Document ingested successfully.",
            filename=file.filename,
            chunks_added=chunks,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(body: QueryRequest):
    """Ask a question. Returns an answer grounded in your uploaded documents."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = answer_question(body.question, top_k=body.top_k)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/sources", response_model=SourcesResponse, tags=["Documents"])
def sources():
    """List all documents currently indexed."""
    srcs = list_sources()
    return SourcesResponse(sources=srcs, total=len(srcs))

@app.delete("/source/{filename}", response_model=DeleteResponse, tags=["Documents"])
def delete(filename: str):
    """Remove all chunks for a given document from the vector store."""
    try:
        delete_source(filename)
        logger.info(f"Deleted '{filename}' from vector store")
        return DeleteResponse(message="Document deleted successfully.", filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
