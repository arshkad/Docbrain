"""
DocBrain — Universal Business Document Intelligence API
A RAG-powered FastAPI service for querying and summarizing business documents.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from app.routers import documents, query, collections
from app.database import chroma_client
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ChromaDB on startup."""
    print("🧠 DocBrain starting up...")
    chroma_client.heartbeat()
    print("✅ Vector store connected")
    yield
    print("👋 DocBrain shutting down")


app = FastAPI(
    title="DocBrain API",
    description="""
## Universal Business Document Intelligence

Upload any business document and instantly get:
- **Semantic Q&A** — ask questions in plain English
- **Smart Summaries** — executive, detailed, or bullet-point
- **Key Insights** — auto-extracted action items, risks, dates
- **Multi-doc Search** — query across your entire document library

### Workflow
1. Create a **collection** (e.g. "contracts-2024", "hr-policies")
2. **Upload** PDFs, text files, or paste raw text
3. **Query** with natural language
4. Get grounded, cited answers

Built with FastAPI + ChromaDB + Claude.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(collections.router, prefix="/collections", tags=["Collections"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(query.router, prefix="/query", tags=["Query & Insights"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "DocBrain API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    try:
        chroma_client.heartbeat()
        return {"status": "healthy", "vector_store": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {e}")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
