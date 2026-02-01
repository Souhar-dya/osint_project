"""
Batch analysis endpoint for multiple texts
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import time
import asyncio
from typing import List

from app.models.schemas import (
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    AnalysisRequest,
    AnalysisResponse
)
from app.database import get_db
from app.routers.analysis import analyze_text

router = APIRouter()


@router.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analyze(
    request: BatchAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze multiple texts in batch.
    Limited to 50 texts per request.
    """
    start_time = time.time()
    
    if len(request.texts) > 50:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 50 texts per batch request"
        )
    
    results: List[AnalysisResponse] = []
    
    for text in request.texts:
        try:
            single_request = AnalysisRequest(
                text=text,
                source=request.source,
                anonymize=request.anonymize
            )
            result = await analyze_text(single_request, db)
            results.append(result)
        except Exception as e:
            # Log error but continue with other texts
            print(f"Error processing text: {str(e)}")
            continue
    
    total_time = int((time.time() - start_time) * 1000)
    
    return BatchAnalysisResponse(
        results=results,
        total_processed=len(results),
        total_time_ms=total_time
    )
