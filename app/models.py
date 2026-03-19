from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class SourceChunk(BaseModel):
    content: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    question: str

class SourcesResponse(BaseModel):
    sources: List[str]
    total: int

class DeleteResponse(BaseModel):
    message: str
    filename: str
