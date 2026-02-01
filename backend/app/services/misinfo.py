"""
Misinformation Detection Module
Uses custom DeBERTa model (89% F1 accuracy)
Simple, clean implementation - let the model do its work
"""

from typing import Optional
import logging
from app.models.schemas import MisinfoResult
from app.services.model_loader import predict_misinfo

logger = logging.getLogger(__name__)


def detect_misinformation(text: str) -> MisinfoResult:
    """
    Detect potential misinformation in text.
    Uses custom DeBERTa model (89% F1) directly.
    """
    try:
        # Use the trained model directly
        result = predict_misinfo(text)
        
        # Model returns: label (fake/real), confidence, scores
        if result['label'] == 'fake':
            risk_score = result['confidence']
        else:
            # If model says "real", risk is inverse of confidence
            risk_score = 1 - result['confidence']
        
        # Determine risk level based on score
        if risk_score >= 0.65:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Simple trigger based on model prediction
        triggers = []
        if result['label'] == 'fake' and result['confidence'] > 0.6:
            triggers.append("AI detected potential misinformation")
        
        return MisinfoResult(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            triggers=triggers,
            claim_type=None
        )
        
    except Exception as e:
        logger.error(f"Misinformation detection error: {e}")
        return MisinfoResult(
            risk_score=0.0,
            risk_level="low",
            triggers=[],
            claim_type=None
        )

