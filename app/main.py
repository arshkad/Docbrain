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

