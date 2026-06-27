"""Query, summarization, and insights endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import (
    QueryRequest, QueryResponse,
    SummaryRequest, SummaryResponse,
    InsightsResponse,
    CompareRequest, CompareResponse,
)
from app.llm import rag_query, rag_query_stream, summarize_document, extract_insights, compare_documents

router = APIRouter()

def _history_to_dicts(history):
    """Convert Pydantic ConversationTurn list to plain dicts for the Anthropic SDK."""
    if not history:
        return None
    return [{"role": t.role, "content": t.content} for t in history]


@router.post("/ask", response_model=QueryResponse)
async def ask_question(body: QueryRequest):
    """
    **Ask a question about your documents.**

    Uses RAG (Retrieval-Augmented Generation):
    1. Semantic search finds the most relevant document chunks
    2. Claude generates a grounded answer with citations

    Pass `history` (prior user/assistant turns) to support natural follow-up
    questions like "what about the second one?".

    Great for: contracts Q&A, policy lookups, research, data extraction
    """
    try:
        result = rag_query(
            collection_name=body.collection_name,
            question=body.question,
            top_k=body.top_k,
            doc_filter=body.doc_filter,
            history=_history_to_dicts(body.history),
        )
        return QueryResponse(**result)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")
