import logging
import os
import time

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from app.ingest import get_vectorstore

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based only on the provided context.
If the answer is not in the context, say "I could not find an answer in the provided documents."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

def get_llm():
    return OllamaLLM(
        base_url=OLLAMA_HOST,
        model=OLLAMA_MODEL,
        temperature=0.1,
    )

def answer_question(question: str, top_k: int = 3) -> dict:
    """Retrieve relevant chunks and generate an answer. Returns answer + sources."""
    start = time.time()

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa_chain.invoke({"query": question})

    elapsed = round(time.time() - start, 2)
    logger.info(f"Query answered in {elapsed}s | question='{question}'")

    sources = [
        {
            "content": doc.page_content[:300],
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in result["source_documents"]
    ]

    return {
        "answer": result["result"].strip(),
        "sources": sources,
        "question": question,
    }
