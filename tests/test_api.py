import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)

# ── health checks ──────────────────────────────────────────────
def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

# ── ingest ─────────────────────────────────────────────────────
def test_ingest_unsupported_format():
    r = client.post(
        "/ingest",
        files={"file": ("test.docx", b"fake content", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "supported" in r.json()["detail"].lower()

def test_ingest_txt_success():
    txt_content = b"Puneeth is an AI engineer who builds RAG systems with FastAPI and ChromaDB."
    with patch("app.main.ingest_file", return_value=3):
        r = client.post(
            "/ingest",
            files={"file": ("sample.txt", io.BytesIO(txt_content), "text/plain")},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "sample.txt"
    assert data["chunks_added"] == 3

def test_ingest_empty_question_rejected():
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400

# ── query ──────────────────────────────────────────────────────
def test_query_returns_answer():
    mock_result = {
        "answer": "RAG stands for Retrieval Augmented Generation.",
        "sources": [{"content": "RAG is a technique...", "source": "sample.txt"}],
        "question": "What is RAG?",
    }
    with patch("app.main.answer_question", return_value=mock_result):
        r = client.post("/query", json={"question": "What is RAG?"})
    assert r.status_code == 200
    assert "answer" in r.json()
    assert "sources" in r.json()

# ── sources ────────────────────────────────────────────────────
def test_list_sources():
    with patch("app.main.list_sources", return_value=["doc1.pdf", "doc2.txt"]):
        r = client.get("/sources")
    assert r.status_code == 200
    assert r.json()["total"] == 2

# ── delete ─────────────────────────────────────────────────────
def test_delete_not_found():
    with patch("app.main.delete_source", side_effect=ValueError("No document found")):
        r = client.delete("/source/ghost.pdf")
    assert r.status_code == 404

def test_delete_success():
    with patch("app.main.delete_source", return_value=None):
        r = client.delete("/source/sample.txt")
    assert r.status_code == 200
    assert r.json()["filename"] == "sample.txt"
