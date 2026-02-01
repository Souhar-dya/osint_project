"""
Topic & Narrative Extraction Module
Uses SBERT embeddings + BERTopic for clustering
"""
from typing import List, Optional
import logging
import numpy as np

from app.models.schemas import TopicResult
from app.services.model_loader import get_model

logger = logging.getLogger(__name__)

# Pre-defined narrative topics for OSINT
NARRATIVE_TOPICS = {
    0: {"label": "Politics/Government", "keywords": ["government", "election", "president", "policy", "vote", "congress"]},
    1: {"label": "Economy/Finance", "keywords": ["economy", "market", "stock", "inflation", "jobs", "money", "bank"]},
    2: {"label": "Health/Medical", "keywords": ["health", "covid", "vaccine", "hospital", "doctor", "disease", "medical"]},
    3: {"label": "Technology", "keywords": ["tech", "ai", "software", "app", "digital", "computer", "data"]},
    4: {"label": "Climate/Environment", "keywords": ["climate", "environment", "green", "carbon", "pollution", "weather"]},
    5: {"label": "Social Issues", "keywords": ["rights", "equality", "justice", "protest", "community", "social"]},
    6: {"label": "Security/Conflict", "keywords": ["war", "military", "security", "attack", "defense", "conflict"]},
    7: {"label": "Entertainment/Culture", "keywords": ["movie", "music", "celebrity", "culture", "art", "entertainment"]},
    8: {"label": "Sports", "keywords": ["game", "team", "player", "championship", "score", "sports"]},
    9: {"label": "Science", "keywords": ["research", "study", "scientist", "discovery", "experiment", "science"]}
}


def extract_topics(text: str) -> TopicResult:
    """
    Extract topic/narrative from text using embeddings.
    Uses keyword matching as primary method, with optional BERTopic.
    """
    try:
        embedder = get_model("embedder")
        
        if embedder is None:
            return _keyword_based_topics(text)
        
        # Generate embedding for the text
        text_embedding = embedder.encode(text, convert_to_numpy=True)
        
        # Generate embeddings for topic keywords
        topic_scores = {}
        for topic_id, topic_data in NARRATIVE_TOPICS.items():
            # Create a representative sentence from keywords
            topic_text = " ".join(topic_data["keywords"])
            topic_embedding = embedder.encode(topic_text, convert_to_numpy=True)
            
            # Cosine similarity
            similarity = np.dot(text_embedding, topic_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(topic_embedding)
            )
            topic_scores[topic_id] = similarity
        
        # Get best matching topic
        best_topic_id = max(topic_scores, key=topic_scores.get)
        best_score = topic_scores[best_topic_id]
        
        # Extract keywords from text that match topic
        text_lower = text.lower()
        matched_keywords = [
            kw for kw in NARRATIVE_TOPICS[best_topic_id]["keywords"] 
            if kw in text_lower
        ]
        
        # Add any other significant words
        additional_keywords = _extract_significant_words(text)
        all_keywords = list(set(matched_keywords + additional_keywords))[:5]
        
        return TopicResult(
            topic_id=best_topic_id,
            topic_label=NARRATIVE_TOPICS[best_topic_id]["label"],
            keywords=all_keywords if all_keywords else NARRATIVE_TOPICS[best_topic_id]["keywords"][:3],
            confidence=round(float(best_score), 4)
        )
        
    except Exception as e:
        logger.error(f"Topic extraction error: {e}")
        return _keyword_based_topics(text)


def _keyword_based_topics(text: str) -> TopicResult:
    """Fallback keyword-based topic detection"""
    text_lower = text.lower()
    
    topic_scores = {}
    for topic_id, topic_data in NARRATIVE_TOPICS.items():
        score = sum(1 for kw in topic_data["keywords"] if kw in text_lower)
        topic_scores[topic_id] = score
    
    best_topic_id = max(topic_scores, key=topic_scores.get)
    best_score = topic_scores[best_topic_id]
    
    if best_score == 0:
        # No clear topic match
        return TopicResult(
            topic_id=-1,
            topic_label="General/Uncategorized",
            keywords=_extract_significant_words(text)[:3],
            confidence=0.3
        )
    
    return TopicResult(
        topic_id=best_topic_id,
        topic_label=NARRATIVE_TOPICS[best_topic_id]["label"],
        keywords=NARRATIVE_TOPICS[best_topic_id]["keywords"][:3],
        confidence=min(best_score * 0.15, 0.9)
    )


def _extract_significant_words(text: str, top_n: int = 5) -> List[str]:
    """Extract significant words from text (simple approach)"""
    # Common stop words to filter out
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "and", "but", "or", "if", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they", "user"
    }
    
    # Tokenize and filter
    words = text.lower().split()
    words = [w.strip('.,!?"\'-:;()[]{}') for w in words]
    words = [w for w in words if len(w) > 3 and w not in stop_words and w.isalpha()]
    
    # Count frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]


def get_topic_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for text (used by baseline comparison)"""
    embedder = get_model("embedder")
    if embedder:
        return embedder.encode(text, convert_to_numpy=True)
    return None
