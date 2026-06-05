"""
FastAPI Application — Doc-Intel RAG Platform

Endpoints:
  GET  /                  → Serve the frontend SPA
  GET  /health            → Health check
  POST /api/query         → RAG query (JSON response)
  POST /api/query/stream  → RAG query with SSE streaming
  POST /api/ingest        → Trigger document ingestion pipeline
  GET  /api/status        → Index status from Pinecone
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── App Bootstrap ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Doc-Intel RAG Platform",
    description="Enterprise-Grade Document Intelligence powered by Hybrid Search + LLMs",
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Frontend ────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Request / Response Schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_n: int = Field(default=10, ge=1, le=50, description="Candidates to retrieve")
    top_k: int = Field(default=3, ge=1, le=10, description="Re-ranked results to use")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Dense/sparse balance")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class IngestRequest(BaseModel):
    data_dir: str = Field(default="Data/Input", description="Directory containing input PDFs")


# ── Lazy Pipeline Singleton ────────────────────────────────────────────────────

_pipeline = None

def get_pipeline():
    """Lazily initialize the RAG pipeline (heavy — loads models once)."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing RAG pipeline for the first time...")
        from core.retrieval_and_generation_4 import RAGPipeline
        _pipeline = RAGPipeline()
        logger.info("RAG pipeline ready.")
    return _pipeline


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend HTML file."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Doc-Intel API is running. Frontend not found."}


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "llm_model": settings.llm_model,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Run a RAG query and return the full answer with source citations.
    This is a blocking endpoint — use /api/query/stream for streaming.
    """
    try:
        pipeline = get_pipeline()

        # Run retrieval + rerank in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        contexts = await loop.run_in_executor(
            None,
            lambda: pipeline.retrieve_and_rerank(
                query=request.query,
                retrieve_top_n=request.top_n,
                keep_top_k=request.top_k,
            ),
        )

        if not contexts:
            return QueryResponse(
                answer="I couldn't find any relevant documents to answer your question.",
                sources=[],
                query=request.query,
            )

        # Build prompt and call Ollama (non-streaming)
        prompt = pipeline.build_prompt(request.query, contexts)

        import ollama
        # Speed-tuned generation options (mirrors retrieval_and_generation_4.py)
        options = {
            "temperature": settings.llm_temperature if settings.llm_temperature > 0 else 0.1,
            "num_predict": 300,
            "num_ctx": 2048,
            "top_k": 20,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
        messages = [
            {"role": "system", "content": "You are a helpful, precise, and concise document assistant."},
            {"role": "user", "content": prompt},
        ]

        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(model=settings.llm_model, messages=messages, stream=False, options=options),
        )

        answer = response["message"]["content"]

        sources = [
            {
                "id": c["id"],
                "source_file": c["metadata"].get("source_file", "Unknown"),
                "score": round(c.get("rerank_score", c.get("score", 0)), 4),
                "excerpt": c["metadata"].get("text", "")[:300],
            }
            for c in contexts
        ]

        return QueryResponse(answer=answer, sources=sources, query=request.query)

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_rag_stream(request: QueryRequest):
    """
    Run a RAG query with Server-Sent Events (SSE) streaming for live token output.
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            pipeline = get_pipeline()
            loop = asyncio.get_running_loop()

            # Retrieval (blocking → thread)
            contexts = await loop.run_in_executor(
                None,
                lambda: pipeline.retrieve_and_rerank(
                    query=request.query,
                    retrieve_top_n=request.top_n,
                    keep_top_k=request.top_k,
                ),
            )

            if not contexts:
                payload = json.dumps({"type": "error", "content": "No relevant documents found."})
                yield f"data: {payload}\n\n"
                return

            # Emit sources first so the UI can show them immediately
            sources = [
                {
                    "id": c["id"],
                    "source_file": c["metadata"].get("source_file", "Unknown"),
                    "score": round(c.get("rerank_score", c.get("score", 0)), 4),
                    "excerpt": c["metadata"].get("text", "")[:300],
                }
                for c in contexts
            ]
            yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

            # Build prompt
            prompt = pipeline.build_prompt(request.query, contexts)

            import ollama
            # Speed-tuned generation options (mirrors retrieval_and_generation_4.py)
            options = {
                "temperature": settings.llm_temperature if settings.llm_temperature > 0 else 0.1,
                "num_predict": 300,
                "num_ctx": 2048,
                "top_k": 20,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
            messages = [
                {"role": "system", "content": "You are a helpful, precise, and concise document assistant."},
                {"role": "user", "content": prompt},
            ]

            # Stream tokens from Ollama
            def stream_ollama():
                return ollama.chat(
                    model=settings.llm_model,
                    messages=messages,
                    stream=True,
                    options=options,
                )

            response_iter = await loop.run_in_executor(None, stream_ollama)

            for chunk in response_iter:
                token = chunk["message"]["content"]
                if token:
                    payload = json.dumps({"type": "token", "content": token})
                    yield f"data: {payload}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ingest")
async def ingest_documents(request: IngestRequest):
    """
    Trigger the full ingestion pipeline: PDF → Chunks → Pinecone.
    This is a long-running operation and runs in a background thread.
    """
    try:
        loop = asyncio.get_event_loop()

        def run_ingestion():
            from core.doc_processor_1 import DocumentProcessor
            from core.chunking_2 import DocumentChunker
            from core.embedding_and_indexing_3 import HybridSearchIndexer

            processor = DocumentProcessor(input_dir=request.data_dir)
            processor.process_all()

            chunker = DocumentChunker()
            chunker.chunk_all()

            indexer = HybridSearchIndexer()
            indexer.index_all()

        await loop.run_in_executor(None, run_ingestion)

        return {"status": "success", "message": "Ingestion pipeline completed successfully."}

    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def index_status():
    """Return Pinecone index statistics."""
    try:
        loop = asyncio.get_event_loop()

        def get_stats():
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.pinecone_index_name)
            return index.describe_index_stats()

        stats = await loop.run_in_executor(None, get_stats)

        total_vectors = stats.get("total_vector_count", 0)
        namespaces = stats.get("namespaces", {})

        return {
            "index_name": settings.pinecone_index_name,
            "total_vectors": total_vectors,
            "namespaces": namespaces,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
        }

    except Exception as e:
        logger.error(f"Status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Document Viewer Endpoints ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "Data" / "Output"
CHUNKS_DIR   = PROJECT_ROOT / "Data" / "chunks"
INPUT_DIR    = PROJECT_ROOT / "Data" / "Input"


@app.get("/api/documents")
async def list_documents():
    """
    List all ingested documents.
    Returns metadata for every processed Markdown file in Data/Output,
    cross-referenced with available chunk counts from Data/chunks.
    """
    docs = []

    if not OUTPUT_DIR.exists():
        return {"documents": docs, "total": 0}

    # Count chunks per document stem for display
    chunk_counts: dict[str, int] = {}
    if CHUNKS_DIR.exists():
        for jf in CHUNKS_DIR.glob("*_chunks.jsonl"):
            stem = jf.stem.replace("_chunks", "")
            count = sum(1 for line in jf.open(encoding="utf-8") if line.strip())
            chunk_counts[stem] = count

    for md_file in sorted(OUTPUT_DIR.glob("*.md")):
        stat = md_file.stat()
        stem = md_file.stem
        # Check if original PDF exists
        pdf_exists = (INPUT_DIR / f"{stem}.pdf").exists()
        docs.append({
            "name": stem,
            "filename": md_file.name,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "chunks": chunk_counts.get(stem, 0),
            "has_pdf": pdf_exists,
        })

    return {"documents": docs, "total": len(docs)}


@app.get("/api/documents/{filename:path}")
async def get_document(filename: str):
    """
    Return the full Markdown content of a processed document.
    `filename` should be the .md filename (e.g. 'report.md').
    """
    # Sanitise — only allow filenames, no path traversal
    safe_name = Path(filename).name
    if not safe_name.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are supported.")

    target = OUTPUT_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Document '{safe_name}' not found.")

    content = target.read_text(encoding="utf-8")
    stem = target.stem

    # Chunk count
    chunk_count = 0
    chunk_file = CHUNKS_DIR / f"{stem}_chunks.jsonl"
    if chunk_file.exists():
        chunk_count = sum(1 for line in chunk_file.open(encoding="utf-8") if line.strip())

    return {
        "name": stem,
        "filename": safe_name,
        "content": content,
        "size_bytes": len(content.encode("utf-8")),
        "chunks": chunk_count,
    }
