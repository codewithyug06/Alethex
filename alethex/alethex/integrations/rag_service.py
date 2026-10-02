"""
Universal RAG Microservice API for ALETHEX.
Language-agnostic REST API allowing any application (Node.js, Python, Go, Rust, Java, C#)
to reconcile vector database chunks before calling any LLM.

Usage:
    python -m alethex.integrations.rag_service --port 8080
    Or via CLI:
    alethex rag-service --port 8080
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from alethex.integrations.universal_rag import UniversalRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ALETHEX-RAG-SERVICE] %(message)s")
logger = logging.getLogger("rag-service")

app = FastAPI(
    title="ALETHEX Universal RAG Reconciliation Service",
    description="Language-agnostic RAG middleware. Reconciles retrieved vector chunks, detects contradictions, and filters superseded facts for ANY LLM.",
    version="1.0.0"
)

# Enable CORS for web apps (e.g. Next.js, React, Vue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RAG_LAYER: Optional[UniversalRAG] = None

def get_rag_layer() -> UniversalRAG:
    global _RAG_LAYER
    if _RAG_LAYER is None:
        _RAG_LAYER = UniversalRAG(mode="drop", device="cpu", use_onnx=True)
    return _RAG_LAYER


class RAGChunk(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional chunk identifier")
    text: str = Field(description="Content text of retrieved chunk")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of when fact was written")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary metadata from vector DB")


class FilterRequest(BaseModel):
    chunks: List[Any] = Field(description="List of chunks (strings, objects, or dicts)")
    mode: Optional[str] = Field(default="drop", description="'drop' to exclude invalid chunks, 'annotate' to keep with warnings")


class ContextRequest(BaseModel):
    chunks: List[Any] = Field(description="List of chunks to format into a clean context string")
    separator: Optional[str] = Field(default="\n\n", description="Separator between chunks in context prompt")


@app.post("/v1/rag/filter")
def filter_chunks(req: FilterRequest):
    """
    Filters retrieved chunks from any vector database.
    Excludes or annotates outdated, superseded, or conflicting chunks.
    """
    rag = get_rag_layer()
    clean_chunks = rag.filter(req.chunks, mode=req.mode)
    return {
        "status": "success",
        "mode": req.mode or "drop",
        "input_chunks": len(req.chunks),
        "surviving_chunks": len(clean_chunks),
        "reconciled_chunks": clean_chunks
    }


@app.post("/v1/rag/context")
def build_context(req: ContextRequest):
    """
    Returns a unified, clean, reconciled context string.
    Ready to insert directly into {context} for any LLM prompt (Claude, ChatGPT, Grok, Qwen, Ollama).
    """
    rag = get_rag_layer()
    reconciled_context = rag.get_reconciled_context(req.chunks, separator=req.separator or "\n\n")
    return {
        "status": "success",
        "reconciled_context": reconciled_context
    }


@app.post("/v1/rag/audit")
def audit_chunks(req: FilterRequest):
    """
    Performs full mathematical audit on retrieved chunks:
    Calculates Consistency Index (CI), flags contradiction pairs, and finds stale facts.
    """
    rag = get_rag_layer()
    audit_report = rag.audit(req.chunks)
    return {
        "status": "success",
        "audit": audit_report
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ALETHEX Universal RAG Reconciliation Microservice",
        "backend": "ONNX INT8 (Optimized)",
        "compatible_frameworks": ["LangChain", "LlamaIndex", "Haystack", "Vercel AI SDK", "ChromaDB", "Pinecone", "Qdrant", "Weaviate"]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ALETHEX Universal RAG Reconciliation Microservice")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()

    logger.info(f"Starting Universal RAG Microservice on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
