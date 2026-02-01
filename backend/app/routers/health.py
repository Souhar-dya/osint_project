"""
Health check endpoint
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.schemas import HealthResponse
from app.database import get_db
from app.services.model_loader import models_loaded

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Check API health and model status"""
    
    # Test database connection
    db_connected = True
    try:
        db.execute("SELECT 1")
    except Exception:
        db_connected = False
    
    return HealthResponse(
        status="healthy" if models_loaded() and db_connected else "degraded",
        models_loaded=models_loaded(),
        database_connected=db_connected,
        version="1.0.0"
    )
