import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name="documents",
    )

def ingest_file(file_path: str, filename: str) -> int:
    """Load a file, chunk it, embed it, store in ChromaDB. Returns chunk count."""
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use PDF or TXT.")

    documents = loader.load()

    # Tag every chunk with the original filename so we can filter/delete later
    for doc in documents:
        doc.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text could be extracted from this file.")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    return len(chunks)

def list_sources() -> list[str]:
    """Return unique filenames that have been ingested."""
    vectorstore = get_vectorstore()
    result = vectorstore.get(include=["metadatas"])
    sources = {m.get("source", "unknown") for m in result["metadatas"]}
    return sorted(sources)

def delete_source(filename: str) -> None:
    """Delete all chunks belonging to a filename."""
    vectorstore = get_vectorstore()
    result = vectorstore.get(include=["metadatas"])
    ids_to_delete = [
        result["ids"][i]
        for i, m in enumerate(result["metadatas"])
        if m.get("source") == filename
    ]
    if not ids_to_delete:
        raise ValueError(f"No document found with name: {filename}")
    vectorstore.delete(ids=ids_to_delete)
