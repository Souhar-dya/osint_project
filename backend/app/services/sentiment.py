"""
Sentiment Analysis Module
Uses custom DeBERTa model (95.24% F1 accuracy)
"""
from typing import Dict, Optional
import logging

from app.models.schemas import SentimentResult
from app.services.model_loader import predict_sentiment

logger = logging.getLogger(__name__)

# Emotion keywords for basic emotion detection
EMOTION_KEYWORDS = {
    "joy": ["happy", "excited", "great", "amazing", "wonderful", "love", "awesome", "fantastic"],
    "anger": ["angry", "furious", "outraged", "hate", "disgusting", "terrible", "worst"],
    "fear": ["scared", "afraid", "worried", "terrified", "anxious", "panic", "alarming"],
    "sadness": ["sad", "depressed", "heartbroken", "devastated", "crying", "tragic", "grief"],
    "surprise": ["shocked", "surprised", "unexpected", "unbelievable", "wow", "omg"],
    "disgust": ["disgusting", "gross", "sick", "revolting", "nasty", "vile"]
}


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Analyze sentiment using custom DeBERTa model (95.24% F1).
    Returns label (positive/negative/neutral) and confidence score.
    Also extracts basic emotions from text.
    """
    try:
        # Use the centralized predict_sentiment function
        result = predict_sentiment(text)
        
        # Detect emotions
        emotions = _detect_emotions(text)
        
        return SentimentResult(
            label=result["label"],
            score=round(result["confidence"], 4),
            emotions=emotions
        )
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return _rule_based_sentiment(text)


def _rule_based_sentiment(text: str) -> SentimentResult:
    """Fallback rule-based sentiment analysis"""
    text_lower = text.lower()
    
    positive_words = ["good", "great", "excellent", "amazing", "love", "happy", "best", "awesome"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible", "sad", "angry"]
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return SentimentResult(label="positive", score=0.6 + (pos_count * 0.05), emotions=None)
    elif neg_count > pos_count:
        return SentimentResult(label="negative", score=0.6 + (neg_count * 0.05), emotions=None)
    else:
        return SentimentResult(label="neutral", score=0.5, emotions=None)


def _detect_emotions(text: str) -> Optional[Dict[str, float]]:
    """Detect emotions using keyword matching"""
    text_lower = text.lower()
    emotions = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            emotions[emotion] = min(count * 0.2, 1.0)
    
    return emotions if emotions else None
