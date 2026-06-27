"""Analytics dashboard endpoints — usage stats, trends, and activity feed."""

from fastapi import APIRouter, Query
from app import analytics

router = APIRouter()


@router.get("/summary")
async def summary(days: int = Query(30, ge=1, le=365)):
    """High-level KPIs: total queries, uploads, avg latency, success rate."""
    return analytics.get_summary_stats(days=days)


@router.get("/timeseries")
async def timeseries(days: int = Query(14, ge=1, le=90)):
    """Daily query volume for the trend chart."""
    return {"days": analytics.get_query_volume_timeseries(days=days)}


@router.get("/top-documents")
async def top_documents(limit: int = Query(8, ge=1, le=50)):
    """Most frequently queried/summarized/analyzed documents."""
    return {"documents": analytics.get_top_documents(limit=limit)}
