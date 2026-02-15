"""
Real-Time Credibility Analyzer
Takes crawled news articles + the original claim and uses:
  - Cross-referencing across multiple sources
  - Source trust scoring
  - NLP-based consistency checking (using existing misinfo + stance models)
  - Consensus analysis
to produce a real-time credibility verdict.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from difflib import SequenceMatcher

from app.services.news_crawler import (
    CrawledArticle, TRUSTED_SOURCES, KNOWN_UNRELIABLE_SOURCES, SATIRE_SOURCES
)
from app.services.misinfo import detect_misinformation
from app.services.model_loader import predict_misinfo

logger = logging.getLogger(__name__)


# ============================================================
# CREDIBILITY SIGNALS
# ============================================================

# Sensationalist / clickbait patterns
SENSATIONALIST_PATTERNS = [
    r'\b(breaking|shock|bombshell|explosive)\b',
    r'\byou won\'?t believe\b',
    r'\b(exposed|busted|caught|revealed|leaked)\b',
    r'\b(must see|must read|must watch)\b',
    r'!!+',
    r'\bwake up\b',
    r'\bthey don\'?t want you to know\b',
    r'\b(mainstream media|msm).*lie',
    r'\bcoverup\b',
    r'\b(sheeple|plandemic)\b',
]

# Credibility boosting phrases
CREDIBILITY_PHRASES = [
    r'\baccording to\b.*\b(study|research|report|data|official)\b',
    r'\bpeer[- ]reviewed\b',
    r'\b(published in|journal of)\b',
    r'\b(university|institute|laboratory) (of|for)\b',
    r'\b(official|government) (statement|report|data)\b',
    r'\b(reuters|associated press|ap news)\b',
]


def analyze_credibility(
    claim: str,
    articles: List[CrawledArticle],
    use_model: bool = True
) -> Dict[str, Any]:
    """
    Analyze the credibility of a claim based on crawled articles.
    
    Steps:
    1. Source diversity & trust analysis
    2. Content consistency across sources
    3. Sensationalism detection on the claim
    4. AI model-based misinfo detection
    5. Cross-reference scoring
    6. Final verdict
    
    Returns a comprehensive credibility report.
    """
    if not articles:
        return _no_articles_result(claim)
    
    # === 1. Source Analysis ===
    source_analysis = _analyze_sources(articles)
    
    # === 2. Content Consistency ===
    consistency = _check_consistency(claim, articles)
    
    # === 3. Sensationalism Check ===
    sensationalism = _check_sensationalism(claim)
    
    # === 4. AI Model Misinfo Check (on the original claim) ===
    model_result = None
    if use_model:
        try:
            model_result = predict_misinfo(claim)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
    
    # === 5. Cross-Reference Score ===
    cross_ref = _cross_reference_score(claim, articles)
    
    # === 6. Compute Final Credibility Score ===
    credibility_score = _compute_final_score(
        source_analysis, consistency, sensationalism,
        model_result, cross_ref, len(articles)
    )
    
    # === 7. Generate Verdict ===
    verdict = _generate_verdict(credibility_score)
    
    # === 8. Build supporting/contradicting article lists ===
    supporting = []
    contradicting = []
    for article in articles:
        combined = article.title + " " + article.snippet
        sim = _text_similarity(claim, combined)
        entry = {
            "title": article.title,
            "url": article.url,
            "source": article.source,
            "source_domain": article.source_domain,
            "similarity": round(sim, 3),
            "published_date": article.published_date
        }
        domain = article.source_domain.lower()
        is_trusted = any(t in domain for t in TRUSTED_SOURCES)
        is_unreliable = any(u in domain for u in KNOWN_UNRELIABLE_SOURCES)

        # Articles that match on key terms are supporting evidence
        # Low similarity doesn't mean contradiction — it may just mean
        # the headline was worded differently
        if is_unreliable:
            contradicting.append(entry)
        elif sim > 0.08 and is_trusted:
            supporting.append(entry)
        elif sim > 0.05:
            supporting.append(entry)
        # Only mark as contradicting if from an unreliable source
        # or very low relevance — "no match" != "contradiction"
    
    return {
        "claim": claim,
        "credibility_score": round(credibility_score, 4),
        "verdict": verdict["label"],
        "verdict_explanation": verdict["explanation"],
        "confidence": round(verdict["confidence"], 4),
        "risk_level": verdict["risk_level"],
        
        # Detailed breakdown
        "source_analysis": {
            "total_sources": source_analysis["total"],
            "trusted_sources": source_analysis["trusted_count"],
            "unreliable_sources": source_analysis["unreliable_count"],
            "satire_sources": source_analysis["satire_count"],
            "unique_domains": source_analysis["unique_domains"],
            "source_diversity_score": round(source_analysis["diversity_score"], 3),
        },
        "consistency_analysis": {
            "avg_similarity": round(consistency["avg_similarity"], 3),
            "max_similarity": round(consistency["max_similarity"], 3),
            "consistent_sources": consistency["consistent_count"],
            "narrative_consensus": consistency["consensus"],
        },
        "sensationalism": {
            "score": round(sensationalism["score"], 3),
            "triggers": sensationalism["triggers"],
            "is_sensational": sensationalism["is_sensational"],
        },
        "model_prediction": {
            "label": model_result["label"] if model_result else "unavailable",
            "confidence": round(model_result["confidence"], 4) if model_result else 0.0,
        } if model_result or True else {},
        "cross_reference": {
            "score": round(cross_ref["score"], 3),
            "matching_headlines": cross_ref["matching_headlines"],
            "coverage_breadth": cross_ref["coverage_breadth"],
        },
        
        # Article lists
        "supporting_articles": supporting[:10],
        "contradicting_articles": contradicting[:5],
        
        "articles_analyzed": len(articles),
        "timestamp": datetime.utcnow().isoformat()
    }


def _no_articles_result(claim: str) -> Dict[str, Any]:
    """Result when no articles are found"""
    return {
        "claim": claim,
        "credibility_score": 0.3,
        "verdict": "UNVERIFIABLE",
        "verdict_explanation": "No related news articles found. The claim cannot be verified against current news coverage.",
        "confidence": 0.2,
        "risk_level": "medium",
        "source_analysis": {"total_sources": 0, "trusted_sources": 0, "unreliable_sources": 0,
                            "satire_sources": 0, "unique_domains": 0, "source_diversity_score": 0},
        "consistency_analysis": {"avg_similarity": 0, "max_similarity": 0,
                                 "consistent_sources": 0, "narrative_consensus": "none"},
        "sensationalism": {"score": 0, "triggers": [], "is_sensational": False},
        "model_prediction": {"label": "unavailable", "confidence": 0},
        "cross_reference": {"score": 0, "matching_headlines": 0, "coverage_breadth": "none"},
        "supporting_articles": [],
        "contradicting_articles": [],
        "articles_analyzed": 0,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================
# COMPONENT ANALYZERS
# ============================================================

def _analyze_sources(articles: List[CrawledArticle]) -> Dict[str, Any]:
    """Analyze the diversity and trustworthiness of sources"""
    domains = set()
    trusted_count = 0
    unreliable_count = 0
    satire_count = 0
    
    for article in articles:
        domain = article.source_domain.lower()
        domains.add(domain)
        
        if any(t in domain for t in TRUSTED_SOURCES):
            trusted_count += 1
        if any(u in domain for u in KNOWN_UNRELIABLE_SOURCES):
            unreliable_count += 1
        if any(s in domain for s in SATIRE_SOURCES):
            satire_count += 1
    
    total = len(articles)
    unique = len(domains)
    
    # Diversity: more unique domains = better
    diversity_score = min(unique / max(total, 1), 1.0) * min(unique / 3.0, 1.0)
    
    # Boost if trusted sources present
    if trusted_count > 0:
        diversity_score = min(diversity_score + 0.2, 1.0)
    
    return {
        "total": total,
        "trusted_count": trusted_count,
        "unreliable_count": unreliable_count,
        "satire_count": satire_count,
        "unique_domains": unique,
        "diversity_score": diversity_score
    }


def _check_consistency(claim: str, articles: List[CrawledArticle]) -> Dict[str, Any]:
    """Check how consistent the claim is with crawled articles"""
    similarities = []
    
    for article in articles:
        combined = article.title + " " + article.snippet
        sim = _text_similarity(claim, combined)
        similarities.append(sim)
    
    if not similarities:
        return {"avg_similarity": 0, "max_similarity": 0,
                "consistent_count": 0, "consensus": "none"}
    
    avg_sim = sum(similarities) / len(similarities)
    max_sim = max(similarities)
    # Lowered threshold: social media text vs headlines often have low
    # raw similarity even when discussing the same event
    consistent = sum(1 for s in similarities if s > 0.08)
    
    if avg_sim > 0.15 and consistent >= 3:
        consensus = "strong"
    elif avg_sim > 0.08 and consistent >= 2:
        consensus = "moderate"
    elif consistent >= 1:
        consensus = "weak"
    else:
        consensus = "none"
    
    return {
        "avg_similarity": avg_sim,
        "max_similarity": max_sim,
        "consistent_count": consistent,
        "consensus": consensus
    }


def _check_sensationalism(text: str) -> Dict[str, Any]:
    """Detect sensationalist / clickbait language"""
    text_lower = text.lower()
    triggers = []
    
    for pattern in SENSATIONALIST_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            triggers.append(pattern.replace(r'\b', '').replace('\\', ''))
    
    # Also check for ALL CAPS words (3+ chars)
    # But exclude proper names — Indian politicians, organizations,
    # abbreviations like "BJP", "AAP", "NDTV" are often typed in caps
    caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    # Filter out words that look like names/abbreviations (<=8 chars usually)
    non_name_caps = [w for w in caps_words if len(w) > 8]
    if len(non_name_caps) > 2 or len(caps_words) > 5:
        triggers.append("excessive_caps")
    
    # Exclamation marks
    if text.count('!') > 2:
        triggers.append("excessive_exclamation")
    
    score = min(len(triggers) * 0.2, 1.0)
    
    # Check for credibility phrases (they reduce sensationalism)
    for pattern in CREDIBILITY_PHRASES:
        if re.search(pattern, text_lower, re.IGNORECASE):
            score = max(score - 0.15, 0.0)
    
    return {
        "score": score,
        "triggers": triggers[:5],
        "is_sensational": score > 0.4
    }


def _cross_reference_score(claim: str, articles: List[CrawledArticle]) -> Dict[str, Any]:
    """
    Cross-reference: how many independent sources report on the same topic?
    Uses named entity matching (proper nouns) + keyword overlap.
    More independent coverage = more likely to be real news.
    """
    # Extract significant words from claim (skip stop words)
    claim_words_raw = re.findall(r'\w{3,}', claim)
    claim_keywords = set(w.lower() for w in claim_words_raw if w.lower() not in _SIM_STOP_WORDS)
    # Extract proper nouns from claim (named entities)
    claim_proper = set(w.lower() for w in claim_words_raw if w[0].isupper() and len(w) > 2
                       and w.lower() not in _SIM_STOP_WORDS)

    matching_headlines = 0
    domains_reporting = set()

    for article in articles:
        title_words_raw = re.findall(r'\w{3,}', article.title)
        title_keywords = set(w.lower() for w in title_words_raw if w.lower() not in _SIM_STOP_WORDS)
        title_proper = set(w.lower() for w in title_words_raw if w[0].isupper() and len(w) > 2
                           and w.lower() not in _SIM_STOP_WORDS)

        # Match if: >=2 proper nouns match, OR >=1 proper noun + some keywords,
        # OR >=15% keyword overlap
        proper_matches = len(claim_proper & title_proper)
        keyword_overlap = len(claim_keywords & title_keywords) / max(len(claim_keywords), 1)

        if (proper_matches >= 2
                or (proper_matches >= 1 and keyword_overlap > 0.08)
                or keyword_overlap > 0.15):
            matching_headlines += 1
            domains_reporting.add(article.source_domain)

    # Coverage breadth
    breadth_count = len(domains_reporting)
    if breadth_count >= 5:
        coverage = "widespread"
    elif breadth_count >= 3:
        coverage = "moderate"
    elif breadth_count >= 1:
        coverage = "limited"
    else:
        coverage = "none"

    # Score: more independent sources = higher score
    # Also factor in how many headlines actually matched
    domain_score = min(breadth_count / 4.0, 1.0)
    headline_score = min(matching_headlines / 3.0, 1.0)
    score = (domain_score * 0.6) + (headline_score * 0.4)

    return {
        "score": score,
        "matching_headlines": matching_headlines,
        "domains_reporting": list(domains_reporting)[:10],
        "coverage_breadth": coverage
    }


def _compute_final_score(
    source_analysis: Dict,
    consistency: Dict,
    sensationalism: Dict,
    model_result: Optional[Dict],
    cross_ref: Dict,
    article_count: int
) -> float:
    """
    Compute weighted credibility score (0 = certainly fake, 1 = certainly real).

    Key principle: REAL-WORLD EVIDENCE (crawler results) should outweigh the
    AI model.  The DeBERTa misinfo model was trained on news-article style text
    and often mislabels social-media posts (informal, emotional, opinionated)
    as "fake".  When the crawler finds multiple real articles from trusted
    sources confirming the event, that is far stronger evidence than the model.

    Weights (rebalanced):
    - Cross-reference (35%): Multiple independent sources confirming
    - Source trust (25%): Quality of reporting sources
    - Consistency (20%): How well claim matches found articles
    - Sensationalism (10%): Penalty for clickbait language
    - AI Model (10%): DeBERTa prediction — tie-breaker only
    """
    score = 0.5  # Start neutral

    # ------- Cross-reference contribution (35%) -------
    # This is the most important signal: did real sources cover this?
    cross_ref_score = cross_ref["score"]
    score += (cross_ref_score - 0.3) * 0.35  # baseline lowered so any matches boost strongly

    # ------- Source trust contribution (25%) -------
    trusted_ratio = source_analysis["trusted_count"] / max(source_analysis["total"], 1)
    unreliable_ratio = source_analysis["unreliable_count"] / max(source_analysis["total"], 1)
    source_score = trusted_ratio - unreliable_ratio
    score += source_score * 0.25

    # Extra boost when trusted sources actually exist
    if source_analysis["trusted_count"] >= 1:
        score += 0.08
    if source_analysis["trusted_count"] >= 3:
        score += 0.05

    # ------- Consistency contribution (20%) -------
    consistency_score = consistency["avg_similarity"]
    score += (consistency_score - 0.15) * 0.20

    # ------- Sensationalism penalty (10%) -------
    if sensationalism["is_sensational"]:
        score -= sensationalism["score"] * 0.10

    # ------- AI Model contribution (10%) — tie-breaker -------
    # The model is good at catching completely fabricated stories, but
    # has high false-positive rate on social media text.  So we:
    #   - Limit its influence to 10%
    #   - Use a symmetric formula so it can't dominate
    if model_result:
        if model_result["label"] == "fake":
            model_signal = 1.0 - model_result["confidence"]  # low → 0, high → 0
        else:
            model_signal = model_result["confidence"]  # high → 1
        # model_signal is in [0,1], center at 0.5
        score += (model_signal - 0.5) * 0.10

    # ------- Coverage volume bonus -------
    if article_count >= 8:
        score += 0.06
    elif article_count >= 5:
        score += 0.04
    elif article_count >= 3:
        score += 0.02

    # ------- Strong override: matching headlines from real sources -------
    # If cross-ref found >=3 matching headlines from trusted sources,
    # the event almost certainly happened — floor the score
    if (cross_ref["matching_headlines"] >= 3
            and source_analysis["trusted_count"] >= 1):
        score = max(score, 0.65)
    elif (cross_ref["matching_headlines"] >= 2
          and source_analysis["trusted_count"] >= 1):
        score = max(score, 0.55)
    elif cross_ref["matching_headlines"] >= 2:
        score = max(score, 0.50)
    elif cross_ref["matching_headlines"] >= 1:
        # Even 1 matching headline means the event is being reported
        score = max(score, 0.45)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def _generate_verdict(credibility_score: float) -> Dict[str, Any]:
    """Generate human-readable verdict from credibility score"""
    if credibility_score >= 0.75:
        return {
            "label": "LIKELY_AUTHENTIC",
            "explanation": "Multiple trusted sources corroborate this claim. High confidence in authenticity.",
            "confidence": credibility_score,
            "risk_level": "low"
        }
    elif credibility_score >= 0.55:
        return {
            "label": "POSSIBLY_AUTHENTIC",
            "explanation": "Some supporting evidence found, but not enough for full verification. Exercise caution.",
            "confidence": credibility_score,
            "risk_level": "low"
        }
    elif credibility_score >= 0.40:
        return {
            "label": "UNVERIFIED",
            "explanation": "Insufficient evidence to confirm or deny this claim. Treat with skepticism.",
            "confidence": credibility_score,
            "risk_level": "medium"
        }
    elif credibility_score >= 0.25:
        return {
            "label": "POSSIBLY_FAKE",
            "explanation": "Limited or contradictory evidence. Multiple signals suggest potential misinformation.",
            "confidence": 1 - credibility_score,
            "risk_level": "high"
        }
    else:
        return {
            "label": "LIKELY_FAKE",
            "explanation": "Strong signals indicate this is likely misinformation. No credible sources support this claim.",
            "confidence": 1 - credibility_score,
            "risk_level": "high"
        }


# Common filler words that don't help similarity matching
_SIM_STOP_WORDS = {
    "the", "is", "are", "was", "were", "been", "being", "have", "has", "had",
    "does", "did", "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "this", "that", "these", "those", "and", "but", "for",
    "not", "you", "all", "any", "her", "his", "its", "our", "they", "who",
    "him", "she", "with", "from", "into", "just", "also", "than", "too",
    "very", "come", "came", "most", "more", "some", "much", "many", "even",
    "about", "over", "such", "here", "there", "when", "what", "which",
    "how", "why", "where", "each", "both", "few", "own", "same", "been",
    "video", "watch", "see", "like", "says", "said", "new", "news",
    # Emotional / sensationalist words that shouldn't drive matching
    "shocking", "shameful", "disgusting", "outrageous", "horrifying",
    "terrible", "horrible", "appalling", "unbelievable", "unacceptable",
    "breaking", "bombshell", "explosive", "sensational", "incredible",
    "amazing", "awesome", "epic", "savage", "brutal", "insane",
    "please", "share", "retweet", "viral", "trending", "thread",
    "urgent", "important", "attention", "everyone", "friends",
    "good", "bad", "best", "worst", "great", "poor",
    "destroying", "ruining", "failing", "betraying", "looting",
    "corrupt", "corruption", "against", "only", "priority", "focus",
}


def _text_similarity(text1: str, text2: str) -> float:
    """
    Compute text similarity using weighted keyword overlap.
    Optimized for comparing social media text against news headlines.
    Gives extra weight to proper nouns / named entities.
    """
    # Extract words (3+ chars)
    words1_raw = re.findall(r'\w{3,}', text1)
    words2_raw = re.findall(r'\w{3,}', text2)

    if not words1_raw or not words2_raw:
        return 0.0

    # Filter stop words but keep originals for proper noun detection
    words1_filtered = [w for w in words1_raw if w.lower() not in _SIM_STOP_WORDS]
    words2_filtered = [w for w in words2_raw if w.lower() not in _SIM_STOP_WORDS]

    set1 = set(w.lower() for w in words1_filtered)
    set2 = set(w.lower() for w in words2_filtered)

    if not set1 or not set2:
        return 0.0

    # Basic Jaccard overlap
    intersection = set1 & set2
    union = set1 | set2
    jaccard = len(intersection) / len(union) if union else 0.0

    # Bonus for proper noun / named entity overlap
    proper1 = set(w.lower() for w in words1_filtered if w[0].isupper() and len(w) > 2)
    proper2 = set(w.lower() for w in words2_filtered if w[0].isupper() and len(w) > 2)
    proper_overlap = proper1 & proper2

    # Proper noun overlap scored as fraction of smaller set
    if proper1 and proper2:
        proper_score = len(proper_overlap) / min(len(proper1), len(proper2))
    else:
        proper_score = 0.0

    # Combined: 60% keyword overlap + 40% named entity overlap
    # Named entities are crucial — if "Rahul Gandhi" appears in both, very relevant
    combined = (jaccard * 0.6) + (proper_score * 0.4)

    return min(combined, 1.0)
