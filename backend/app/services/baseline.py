"""
Baseline Comparison Module (RESEARCH NOVELTY)
Compares social media narratives against event-grounded baselines from GDELT and FEVER.
This module measures "narrative distortion" - how far a claim deviates from verified facts.
"""
from typing import Optional, List, Dict
import logging
import numpy as np
import json
from datetime import datetime, timedelta

from app.models.schemas import BaselineComparison, TopicResult
from app.services.model_loader import get_model

logger = logging.getLogger(__name__)

# Cache for baseline events (in production, use Redis/PostgreSQL)
_baseline_cache: Dict[str, dict] = {}


class BaselineEvent:
    """Represents a verified event from GDELT or FEVER"""
    def __init__(self, event_id: str, source: str, title: str, 
                 description: str, embedding: Optional[np.ndarray] = None):
        self.event_id = event_id
        self.source = source
        self.title = title
        self.description = description
        self.embedding = embedding


# Sample baseline events (in production, these come from GDELT/FEVER APIs)
SAMPLE_BASELINES = [
    {
        "event_id": "gdelt_climate_2024",
        "source": "gdelt",
        "title": "UN Climate Report 2024",
        "description": "Global temperatures rose 1.1°C above pre-industrial levels. Extreme weather events increased by 40%. Paris Agreement targets remain achievable with immediate action.",
        "topic": "Climate/Environment"
    },
    {
        "event_id": "gdelt_vaccine_2024",
        "source": "gdelt", 
        "title": "WHO Vaccine Safety Report",
        "description": "COVID-19 vaccines have prevented over 14 million deaths globally. Side effects remain rare and mild. No evidence of microchips or tracking devices.",
        "topic": "Health/Medical"
    },
    {
        "event_id": "fever_election_integrity",
        "source": "fever",
        "title": "Election Security Assessment",
        "description": "US elections use multiple layers of security including paper trails, audits, and bipartisan oversight. No evidence of widespread fraud in recent elections.",
        "topic": "Politics/Government"
    },
    {
        "event_id": "gdelt_economy_2024",
        "source": "gdelt",
        "title": "Global Economic Outlook",
        "description": "IMF projects 3.2% global growth. Inflation declining in most economies. Employment rates recovering to pre-pandemic levels in developed nations.",
        "topic": "Economy/Finance"
    },
    {
        "event_id": "fever_5g_safety",
        "source": "fever",
        "title": "5G Network Safety Studies",
        "description": "Multiple studies confirm 5G frequencies are non-ionizing and safe. No evidence linking 5G to health issues or virus transmission.",
        "topic": "Technology"
    },
    {
        "event_id": "gdelt_trafficking_2024",
        "source": "gdelt",
        "title": "Human Trafficking & Exploitation Reports",
        "description": "Human trafficking investigations involve complex legal proceedings. Victim identities are protected by law. Court documents and official investigations provide verified facts. Claims about specific cases should be verified against court records.",
        "topic": "Social Issues"
    },
    {
        "event_id": "gdelt_crime_justice_2024",
        "source": "gdelt",
        "title": "Criminal Justice Proceedings",
        "description": "High-profile criminal cases follow due legal process. Allegations require evidence and due process. Media coverage may simplify or sensationalize complex legal matters. Verified information comes from court filings and official statements.",
        "topic": "Security/Conflict"
    },
    {
        "event_id": "fever_social_rights",
        "source": "fever",
        "title": "Social Rights and Civil Liberties",
        "description": "Social justice issues involve nuanced policy discussions. Protests and movements reflect diverse viewpoints. Verified facts come from official reports, academic research, and established news organizations.",
        "topic": "Social Issues"
    },
    {
        "event_id": "gdelt_science_2024",
        "source": "gdelt",
        "title": "Scientific Research Standards",
        "description": "Scientific claims require peer review and reproducibility. Preliminary findings differ from established consensus. Media often oversimplifies research results.",
        "topic": "Science"
    },
    {
        "event_id": "gdelt_entertainment_2024",
        "source": "gdelt",
        "title": "Entertainment Industry Reports",
        "description": "Celebrity news and entertainment industry claims frequently mix facts with speculation. Verified information comes from official statements, court records, and established entertainment journalists.",
        "topic": "Entertainment/Culture"
    }
]


def compare_baseline(text: str, topics: TopicResult) -> BaselineComparison:
    """
    Compare input text against verified event baselines.
    
    This is the RESEARCH NOVELTY component:
    - Measures semantic distance between social media narratives and verified facts
    - Identifies narrative distortion patterns (exaggeration, contradiction, fabrication)
    - Provides explainable deviation scores
    """
    try:
        embedder = get_model("embedder")
        
        if embedder is None:
            return _rule_based_baseline(text, topics)
        
        # Get text embedding
        text_embedding = embedder.encode(text, convert_to_numpy=True)
        
        # Find relevant baselines for the topic
        relevant_baselines = _get_relevant_baselines(topics.topic_label, embedder)
        
        if not relevant_baselines:
            return BaselineComparison(
                narrative_distance=0.5,
                closest_event=None,
                event_source=None,
                deviation_type="unknown"
            )
        
        # Calculate similarity with each baseline
        similarities = []
        for baseline in relevant_baselines:
            if baseline.embedding is not None:
                similarity = np.dot(text_embedding, baseline.embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(baseline.embedding)
                )
                similarities.append((baseline, similarity))
        
        if not similarities:
            return BaselineComparison(
                narrative_distance=0.5,
                closest_event=None,
                event_source=None,
                deviation_type="unknown"
            )
        
        # Find closest baseline
        closest_baseline, max_similarity = max(similarities, key=lambda x: x[1])
        
        # Reject weak matches — if the best match is too dissimilar,
        # the baseline is not relevant to this text at all
        if max_similarity < 0.25:
            return BaselineComparison(
                narrative_distance=0.0,
                closest_event=None,
                event_source=None,
                deviation_type=None
            )
        
        # Calculate narrative distance (inverse of similarity)
        # 0 = perfectly aligned, 1 = completely distorted
        narrative_distance = 1 - max(0, min(1, (max_similarity + 1) / 2))
        
        # Determine deviation type
        deviation_type = _classify_deviation(text, closest_baseline, narrative_distance)
        
        return BaselineComparison(
            narrative_distance=round(float(narrative_distance), 4),
            closest_event=closest_baseline.title,
            event_source=closest_baseline.source,
            deviation_type=deviation_type
        )
        
    except Exception as e:
        logger.error(f"Baseline comparison error: {e}")
        return BaselineComparison(
            narrative_distance=0.5,
            closest_event=None,
            event_source=None,
            deviation_type=None
        )


def _get_relevant_baselines(topic_label: str, embedder) -> List[BaselineEvent]:
    """Get baseline events relevant to the topic"""
    relevant = []
    
    for baseline_data in SAMPLE_BASELINES:
        # Check if baseline is relevant to topic
        if topic_label.lower() in baseline_data["topic"].lower():
            # Create embedding if not cached
            cache_key = baseline_data["event_id"]
            if cache_key not in _baseline_cache:
                embedding = embedder.encode(
                    baseline_data["description"], 
                    convert_to_numpy=True
                )
                _baseline_cache[cache_key] = {
                    **baseline_data,
                    "embedding": embedding
                }
            
            cached = _baseline_cache[cache_key]
            relevant.append(BaselineEvent(
                event_id=cached["event_id"],
                source=cached["source"],
                title=cached["title"],
                description=cached["description"],
                embedding=cached["embedding"]
            ))
    
    # If no topic match, return all baselines
    if not relevant:
        for baseline_data in SAMPLE_BASELINES:
            cache_key = baseline_data["event_id"]
            if cache_key not in _baseline_cache:
                embedding = embedder.encode(
                    baseline_data["description"],
                    convert_to_numpy=True
                )
                _baseline_cache[cache_key] = {
                    **baseline_data,
                    "embedding": embedding
                }
            
            cached = _baseline_cache[cache_key]
            relevant.append(BaselineEvent(
                event_id=cached["event_id"],
                source=cached["source"],
                title=cached["title"],
                description=cached["description"],
                embedding=cached["embedding"]
            ))
    
    return relevant


def _classify_deviation(text: str, baseline: BaselineEvent, distance: float) -> str:
    """
    Classify the type of narrative deviation.
    
    Types:
    - aligned: Text aligns with verified facts
    - exaggeration: Facts present but overstated
    - contradiction: Directly contradicts baseline
    - fabrication: Claims not supported by any baseline
    """
    text_lower = text.lower()
    baseline_lower = baseline.description.lower()
    
    # Check for contradiction patterns
    contradiction_pairs = [
        ("true", "false"), ("safe", "dangerous"), ("proven", "debunked"),
        ("no evidence", "evidence"), ("confirmed", "denied"),
        ("increase", "decrease"), ("rise", "fall")
    ]
    
    for word1, word2 in contradiction_pairs:
        if word1 in text_lower and word2 in baseline_lower:
            return "contradiction"
        if word2 in text_lower and word1 in baseline_lower:
            return "contradiction"
    
    # Check for exaggeration patterns
    exaggeration_words = ["always", "never", "everyone", "nobody", "all", "none",
                          "massive", "huge", "catastrophic", "explosive"]
    if any(word in text_lower for word in exaggeration_words):
        if distance > 0.3:
            return "exaggeration"
    
    # Determine based on distance
    if distance < 0.3:
        return "aligned"
    elif distance < 0.5:
        return "exaggeration"
    elif distance < 0.7:
        return "selective"  # Selective presentation of facts
    else:
        return "fabrication"


def _rule_based_baseline(text: str, topics: TopicResult) -> BaselineComparison:
    """Fallback rule-based baseline comparison"""
    text_lower = text.lower()
    
    # Check for known false narratives
    false_narratives = [
        ("vaccine", ["microchip", "tracking", "magnetic", "5g"]),
        ("election", ["stolen", "rigged", "fraud"]),
        ("climate", ["hoax", "fake", "conspiracy"]),
        ("covid", ["hoax", "fake", "plandemic"])
    ]
    
    for topic, false_claims in false_narratives:
        if topic in text_lower:
            for claim in false_claims:
                if claim in text_lower:
                    return BaselineComparison(
                        narrative_distance=0.9,
                        closest_event=f"Verified {topic} information",
                        event_source="fact_check",
                        deviation_type="fabrication"
                    )
    
    return BaselineComparison(
        narrative_distance=0.3,
        closest_event=None,
        event_source=None,
        deviation_type="unknown"
    )
