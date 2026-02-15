"""
Explainability Layer
Generates human-readable explanations for all analysis results
Critical for OSINT transparency and user trust
"""
from typing import List
import logging

from app.models.schemas import (
    ExplainabilityOutput,
    SentimentResult,
    TopicResult,
    FramingResult,
    MisinfoResult,
    BaselineComparison
)

logger = logging.getLogger(__name__)


def generate_explanation(
    text: str,
    sentiment: SentimentResult,
    topics: TopicResult,
    framing: FramingResult,
    misinfo: MisinfoResult,
    baseline: BaselineComparison
) -> ExplainabilityOutput:
    """
    Generate comprehensive explanation of analysis results.
    
    This provides:
    1. Overall confidence score
    2. Key phrases that influenced the analysis
    3. Human-readable reasoning
    4. Warning flags
    """
    try:
        # Calculate overall confidence
        confidence = _calculate_overall_confidence(sentiment, topics, framing, misinfo, baseline)
        
        # Extract key phrases
        key_phrases = _extract_key_phrases(text, misinfo, framing)
        
        # Generate reasoning text
        reasoning = _generate_reasoning(sentiment, topics, framing, misinfo, baseline)
        
        # Compile warning flags
        flags = _compile_flags(sentiment, misinfo, baseline, framing)
        
        return ExplainabilityOutput(
            confidence=round(confidence, 4),
            key_phrases=key_phrases[:5],  # Limit to 5
            reasoning=reasoning,
            flags=flags
        )
        
    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        return ExplainabilityOutput(
            confidence=0.5,
            key_phrases=[],
            reasoning="Unable to generate detailed explanation.",
            flags=[]
        )


def _calculate_overall_confidence(
    sentiment: SentimentResult,
    topics: TopicResult,
    framing: FramingResult,
    misinfo: MisinfoResult,
    baseline: BaselineComparison
) -> float:
    """Calculate weighted overall confidence score"""
    
    weights = {
        "sentiment": 0.15,
        "topics": 0.2,
        "framing": 0.2,
        "misinfo": 0.25,
        "baseline": 0.2
    }
    
    # Individual confidence scores
    scores = {
        "sentiment": sentiment.score,
        "topics": topics.confidence,
        "framing": framing.confidence,
        "misinfo": 1 - misinfo.risk_score,  # Invert (high risk = low confidence in authenticity)
        "baseline": 1 - baseline.narrative_distance  # Invert (high distance = low confidence)
    }
    
    weighted_sum = sum(scores[key] * weights[key] for key in weights)
    return min(max(weighted_sum, 0.0), 1.0)


def _extract_key_phrases(text: str, misinfo: MisinfoResult, framing: FramingResult) -> List[str]:
    """Extract phrases that influenced the analysis"""
    key_phrases = []
    
    # Add misinformation triggers
    key_phrases.extend(misinfo.triggers)
    
    # Add propaganda techniques as indicators
    for technique in framing.propaganda_techniques:
        key_phrases.append(f"[{technique}]")
    
    # Extract quoted phrases from text
    import re
    quotes = re.findall(r'"([^"]+)"', text)
    key_phrases.extend(quotes[:2])
    
    # Extract hashtags
    hashtags = re.findall(r'#\w+', text)
    key_phrases.extend(hashtags[:2])
    
    return list(set(key_phrases))


def _generate_reasoning(
    sentiment: SentimentResult,
    topics: TopicResult,
    framing: FramingResult,
    misinfo: MisinfoResult,
    baseline: BaselineComparison
) -> str:
    """Generate actionable, context-aware reasoning that adds insight beyond raw scores"""
    
    parts = []
    
    # Overall assessment first
    if misinfo.risk_level == "high":
        parts.append("⚠️ This content shows significant red flags for misinformation.")
    elif misinfo.risk_level == "medium":
        parts.append("This content contains some concerning elements worth verifying.")
    else:
        parts.append("This content appears to be low-risk based on our analysis.")
    
    # Explain WHY sentiment matters in context
    if sentiment.label == "negative" and sentiment.score > 0.8:
        if misinfo.risk_level != "low":
            parts.append("The highly negative tone may be designed to provoke emotional reactions rather than inform.")
        else:
            parts.append("The negative sentiment reflects the serious nature of the content.")
    elif sentiment.label == "positive" and sentiment.score > 0.8:
        if "health" in topics.topic_label.lower() or "cure" in str(misinfo.triggers).lower():
            parts.append("Be cautious: overly positive health claims often lack evidence.")
    
    # Topic + Framing insight
    if framing.propaganda_techniques:
        techniques = framing.propaganda_techniques[:2]
        if "fear_appeal" in techniques:
            parts.append("This content uses fear-based messaging to influence your reaction.")
        elif "loaded_language" in techniques:
            parts.append("Notice the emotionally charged language - this may bias your perception.")
        elif "appeal_to_authority" in techniques:
            parts.append("Claims authority figures without verifiable sources.")
        else:
            parts.append(f"Rhetorical techniques detected: {', '.join(techniques)}.")
    
    # Misinformation triggers - explain specifically
    if misinfo.triggers:
        trigger_explanations = []
        for trigger in misinfo.triggers[:2]:
            if "urgent" in trigger.lower() or "breaking" in trigger.lower():
                trigger_explanations.append("creates artificial urgency")
            elif "share" in trigger.lower() or "forward" in trigger.lower():
                trigger_explanations.append("pressures viral sharing")
            elif "they don't want" in trigger.lower() or "hidden" in trigger.lower():
                trigger_explanations.append("uses conspiracy framing")
            elif "100%" in trigger.lower() or "guaranteed" in trigger.lower():
                trigger_explanations.append("makes absolute claims without nuance")
        
        if trigger_explanations:
            parts.append(f"Warning signs: {', '.join(trigger_explanations)}.")
    
    # Baseline comparison - the research novelty
    if baseline.closest_event and baseline.deviation_type:
        if baseline.deviation_type == "aligned":
            parts.append(f"✓ Claims align with verified reporting from {baseline.event_source or 'trusted sources'}.")
        elif baseline.deviation_type == "contradiction":
            parts.append(f"✗ This contradicts verified information about: {baseline.closest_event}.")
        elif baseline.deviation_type == "exaggeration":
            parts.append(f"This appears to exaggerate facts from: {baseline.closest_event}.")
        elif baseline.deviation_type == "fabrication":
            parts.append("We found no verified sources supporting these specific claims.")
        elif baseline.deviation_type == "selective":
            parts.append(f"This selectively presents facts related to: {baseline.closest_event}.")
    elif not baseline.closest_event and baseline.narrative_distance == 0.0:
        # No relevant baseline found — don't make up a comparison
        pass
    
    # Actionable recommendation
    if misinfo.risk_level == "high":
        parts.append("Recommendation: Do not share. Verify with multiple trusted sources.")
    elif misinfo.risk_level == "medium":
        parts.append("Recommendation: Check original sources before sharing.")
    else:
        parts.append("This content appears reasonable, but always verify important claims.")
    
    return " ".join(parts)


def _compile_flags(
    sentiment: SentimentResult,
    misinfo: MisinfoResult,
    baseline: BaselineComparison,
    framing: FramingResult
) -> List[str]:
    """Compile warning flags for the content"""
    flags = []
    
    # High misinformation risk
    if misinfo.risk_level == "high":
        flags.append("🚨 HIGH_MISINFO_RISK")
    elif misinfo.risk_level == "medium":
        flags.append("⚠️ MODERATE_MISINFO_RISK")
    
    # Narrative distortion
    if baseline.narrative_distance > 0.7:
        flags.append("🔴 HIGH_NARRATIVE_DISTORTION")
    elif baseline.narrative_distance > 0.5:
        flags.append("🟡 MODERATE_DISTORTION")
    
    # Deviation types
    if baseline.deviation_type == "contradiction":
        flags.append("❌ CONTRADICTS_VERIFIED_FACTS")
    elif baseline.deviation_type == "fabrication":
        flags.append("❌ UNSUPPORTED_CLAIMS")
    
    # Propaganda techniques
    if len(framing.propaganda_techniques) >= 3:
        flags.append("📢 MULTIPLE_PROPAGANDA_TECHNIQUES")
    elif "fear_appeal" in framing.propaganda_techniques:
        flags.append("😨 FEAR_APPEAL_DETECTED")
    elif "loaded_language" in framing.propaganda_techniques:
        flags.append("💬 LOADED_LANGUAGE")
    
    # Strong negative sentiment with high confidence
    if sentiment.label == "negative" and sentiment.score > 0.85:
        flags.append("😠 HIGHLY_NEGATIVE")
    
    return flags[:5]  # Limit flags
