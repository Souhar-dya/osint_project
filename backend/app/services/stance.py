"""
Stance Detection Service
Uses Zero-shot BART for stance classification
Labels: agree, disagree, discuss, unrelated

Features:
- Single text stance analysis
- Reply chain analysis (original vs replies)
- Quote tweet analysis (quoted vs commentary)
"""
import logging
from typing import Dict, List, Any, Optional

from app.services.model_loader import predict_stance, load_all_models, _loaded
from app.models.schemas import StanceResult

logger = logging.getLogger(__name__)


def ensure_models_loaded():
    """Ensure models are loaded before prediction"""
    global _loaded
    from app.services import model_loader
    if not model_loader._loaded:
        logger.info("Models not loaded, loading now...")
        load_all_models()


def analyze_stance(text: str, claim: str = None) -> StanceResult:
    """
    Analyze stance of text towards a claim.
    
    If no claim provided, analyzes general stance/tone of the text.
    
    Args:
        text: The text to analyze
        claim: Optional claim to compare stance against
        
    Returns:
        StanceResult with label, confidence, and scores
    """
    ensure_models_loaded()
    result = predict_stance(text, claim)
    
    return StanceResult(
        label=result["label"],
        confidence=result["confidence"],
        scores=result.get("scores", {})
    )


def analyze_reply_chain(original_text: str, replies: List[str]) -> Dict[str, Any]:
    """
    Analyze stance of replies against the original post
    
    This is useful for understanding how people react to a post:
    - High agree: Post resonates with audience
    - High disagree: Post is controversial or disputed
    - High discuss: Post sparks debate
    - High unrelated: Spam or off-topic replies
    
    Args:
        original_text: The original tweet/post
        replies: List of reply texts
        
    Returns:
        Analysis of reply stances with consensus metrics
    """
    logger.info(f"Analyzing reply chain: {len(replies)} replies to '{original_text[:50]}...'")
    
    # Ensure models are loaded
    ensure_models_loaded()
    
    if not replies:
        return {
            "original": original_text,
            "reply_count": 0,
            "stance_breakdown": {"agree": 0, "disagree": 0, "discuss": 0, "unrelated": 0},
            "consensus": {
                "agreement_ratio": 0,
                "disagreement_ratio": 0,
                "controversial": False,
                "dominant_stance": "none",
                "engagement_quality": "no_replies"
            },
            "replies": []
        }
    
    results = []
    stance_counts = {"agree": 0, "disagree": 0, "discuss": 0, "unrelated": 0}
    
    for i, reply in enumerate(replies):
        try:
            # Format for stance detection: claim vs response
            stance = predict_stance(reply, claim=original_text)
            
            results.append({
                "index": i,
                "reply": reply[:150] + "..." if len(reply) > 150 else reply,
                "stance": stance["label"],
                "confidence": stance["confidence"]
            })
            
            if stance["label"] in stance_counts:
                stance_counts[stance["label"]] += 1
                
        except Exception as e:
            logger.error(f"Error analyzing reply {i}: {e}")
            results.append({
                "index": i,
                "reply": reply[:150] + "..." if len(reply) > 150 else reply,
                "stance": "error",
                "confidence": 0.0
            })
    
    # Calculate consensus metrics
    total = len(replies)
    agree_ratio = stance_counts["agree"] / total
    disagree_ratio = stance_counts["disagree"] / total
    discuss_ratio = stance_counts["discuss"] / total
    unrelated_ratio = stance_counts["unrelated"] / total
    
    # Determine engagement quality
    if unrelated_ratio > 0.5:
        engagement_quality = "spam_heavy"
    elif disagree_ratio > 0.4:
        engagement_quality = "controversial"
    elif agree_ratio > 0.6:
        engagement_quality = "supportive"
    elif discuss_ratio > 0.4:
        engagement_quality = "debated"
    else:
        engagement_quality = "mixed"
    
    # Determine dominant stance
    dominant_stance = max(stance_counts, key=stance_counts.get)
    if stance_counts[dominant_stance] < total * 0.3:
        dominant_stance = "mixed"
    
    consensus = {
        "agreement_ratio": round(agree_ratio, 3),
        "disagreement_ratio": round(disagree_ratio, 3),
        "discuss_ratio": round(discuss_ratio, 3),
        "unrelated_ratio": round(unrelated_ratio, 3),
        "controversial": disagree_ratio > 0.3,
        "dominant_stance": dominant_stance,
        "engagement_quality": engagement_quality,
        "credibility_signal": get_credibility_signal(stance_counts, total)
    }
    
    return {
        "original": original_text,
        "reply_count": total,
        "stance_breakdown": stance_counts,
        "consensus": consensus,
        "replies": results
    }


def get_credibility_signal(stance_counts: Dict[str, int], total: int) -> str:
    """
    Generate a credibility signal based on reply stances
    
    Logic:
    - If many disagree: Claim may be false/misleading
    - If many agree from diverse sources: Claim may be credible
    - If mostly discuss: Topic is complex/debated
    """
    if total == 0:
        return "insufficient_data"
    
    disagree_ratio = stance_counts["disagree"] / total
    agree_ratio = stance_counts["agree"] / total
    
    if disagree_ratio > 0.5:
        return "likely_disputed"
    elif disagree_ratio > 0.3:
        return "some_pushback"
    elif agree_ratio > 0.6:
        return "generally_accepted"
    else:
        return "mixed_reception"


def analyze_quote_tweet(quoted_text: str, commentary: str) -> Dict[str, Any]:
    """
    Analyze relationship between quoted content and commentary
    
    Quote tweets are interesting because people often:
    - Agree and amplify the message
    - Disagree and critique/debunk
    - Add context or discussion
    - Use for unrelated purposes (ratio, etc.)
    
    Args:
        quoted_text: The original quoted tweet
        commentary: The user's added commentary
        
    Returns:
        Analysis of the quote tweet relationship
    """
    logger.info(f"Analyzing quote tweet relationship")
    
    # Get stance of commentary towards quoted content
    stance = predict_stance(commentary, claim=quoted_text)
    
    # Determine the type of quote tweet
    quote_type = determine_quote_type(stance["label"], stance["confidence"])
    
    # Generate explanation
    explanation = generate_quote_explanation(quote_type, stance)
    
    return {
        "quoted_text": quoted_text[:200] + "..." if len(quoted_text) > 200 else quoted_text,
        "commentary": commentary[:200] + "..." if len(commentary) > 200 else commentary,
        "stance": stance["label"],
        "confidence": stance["confidence"],
        "quote_type": quote_type,
        "explanation": explanation,
        "credibility_modifier": get_credibility_modifier(quote_type)
    }


def determine_quote_type(stance: str, confidence: float) -> str:
    """Determine the type of quote tweet based on stance"""
    if confidence < 0.4:
        return "ambiguous"
    
    type_mapping = {
        "agree": "amplification",      # Sharing and endorsing
        "disagree": "critique",        # Challenging or debunking
        "discuss": "commentary",       # Adding context or discussion
        "unrelated": "tangential"      # Off-topic or ratio attempt
    }
    
    return type_mapping.get(stance, "unknown")


def generate_quote_explanation(quote_type: str, stance: Dict) -> str:
    """Generate human-readable explanation"""
    explanations = {
        "amplification": "The user appears to endorse and share this content approvingly.",
        "critique": "The user appears to challenge or disagree with this content.",
        "commentary": "The user is adding context or discussing this content without clear endorsement.",
        "tangential": "The user's commentary doesn't directly relate to the quoted content.",
        "ambiguous": "The relationship between the quote and commentary is unclear."
    }
    
    return explanations.get(quote_type, "Unable to determine relationship.")


def get_credibility_modifier(quote_type: str) -> float:
    """
    Get credibility modifier based on quote type
    
    Returns a modifier that can be applied to misinfo score:
    - Positive: Increases concern (likely spreading misinfo)
    - Negative: Decreases concern (likely debunking)
    - Zero: Neutral
    """
    modifiers = {
        "amplification": 0.1,   # Slight concern if amplifying
        "critique": -0.15,      # Positive signal if critiquing
        "commentary": 0.0,      # Neutral
        "tangential": 0.0,      # Neutral
        "ambiguous": 0.0        # Neutral
    }
    
    return modifiers.get(quote_type, 0.0)


def analyze_thread_credibility(posts: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze a full thread for credibility signals
    
    Args:
        posts: List of dicts with 'text' and optionally 'author', 'timestamp'
        
    Returns:
        Thread-level credibility analysis
    """
    if not posts:
        return {"error": "No posts provided"}
    
    # First post is the original claim
    original = posts[0]
    replies = posts[1:]
    
    # Analyze reply chain
    reply_texts = [p.get("text", "") for p in replies if p.get("text")]
    chain_analysis = analyze_reply_chain(original.get("text", ""), reply_texts)
    
    # Calculate thread credibility score
    consensus = chain_analysis["consensus"]
    
    # Base credibility on how replies react
    if consensus["engagement_quality"] == "controversial":
        credibility = 0.4  # Disputed content
        assessment = "This thread's original claim is being disputed by many replies"
    elif consensus["credibility_signal"] == "likely_disputed":
        credibility = 0.3
        assessment = "Multiple replies contradict or debunk the original claim"
    elif consensus["credibility_signal"] == "generally_accepted":
        credibility = 0.7
        assessment = "Replies generally support or accept the original claim"
    else:
        credibility = 0.5
        assessment = "Mixed reactions - verify the claim independently"
    
    return {
        "thread_length": len(posts),
        "original_claim": original.get("text", "")[:200],
        "reply_analysis": chain_analysis,
        "thread_credibility": round(credibility, 2),
        "assessment": assessment,
        "recommendation": "Cross-reference with trusted sources" if credibility < 0.6 else "Appears to have community support"
    }

