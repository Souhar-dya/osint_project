"""
FastAPI Backend for AI-Based OSINT Monitoring System
Main entry point with all routes and middleware
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.routers import analysis, health, batch, crawler
from app.database import init_db
from app.services.model_loader import load_all_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - load models on startup"""
    logger.info("Loading AI models...")
    load_all_models()
    init_db()
    logger.info("Models loaded successfully!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="OSINT Monitoring API",
    description="AI-powered social media analysis for sentiment, narratives, misinformation, and framing detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chrome extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(batch.router, prefix="/api", tags=["Batch"])
app.include_router(crawler.router, prefix="/api", tags=["Crawler"])


@app.get("/")
async def root():
    return {
        "message": "OSINT Monitoring API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "batch": "/api/batch",
            "crawler_search": "/api/crawler/search",
            "crawler_quick_check": "/api/crawler/quick-check",
            "crawler_sources": "/api/crawler/sources",
            "health": "/health"
        }
    }
