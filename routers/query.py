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