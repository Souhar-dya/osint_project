"""
Claim Verification Service
Verifies claims against fact-check sources and known misinformation patterns

Features:
- Local knowledge base of common misinformation
- Google Fact Check API integration (optional)
- Stance-based verification against multiple sources
"""
import logging
import httpx
import re
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher

from app.services.model_loader import predict_stance

logger = logging.getLogger(__name__)

# ============================================================
# KNOWN MISINFORMATION DATABASE
# ============================================================
# Pre-built knowledge base of debunked claims and misinformation

KNOWN_MISINFORMATION = {
    "5g_covid": {
        "patterns": [
            r"5g.*(cause|spread|create).*covid",
            r"5g.*(corona|virus|pandemic)",
            r"5g.*tower.*(dangerous|radiation|health)",
            r"5g.*kill",
        ],
        "keywords": ["5g", "tower", "radiation", "covid", "virus", "dangerous"],
        "verdict": "FALSE",
        "explanation": "No scientific evidence links 5G technology to COVID-19 or health issues. 5G uses non-ionizing radiation similar to 4G, WiFi, and radio waves.",
        "sources": ["WHO", "CDC", "IEEE", "Multiple peer-reviewed studies"],
        "confidence": 0.95
    },
    "vaccine_microchip": {
        "patterns": [
            r"vaccine.*(microchip|chip|track|gps)",
            r"(bill gates|gates).*(microchip|chip|track)",
            r"inject.*chip",
            r"vaccine.*nano.*bot",
        ],
        "keywords": ["vaccine", "microchip", "chip", "gates", "track", "nano"],
        "verdict": "FALSE",
        "explanation": "Vaccines do not contain microchips or tracking devices. This has been thoroughly debunked by multiple fact-checkers and scientists.",
        "sources": ["Reuters", "Snopes", "FactCheck.org", "CDC"],
        "confidence": 0.98
    },
    "covid_cure_bleach": {
        "patterns": [
            r"(bleach|chlorine|disinfectant).*(cure|treat|kill).*covid",
            r"drink.*(bleach|chlorine|disinfectant)",
            r"inject.*(bleach|disinfectant)",
        ],
        "keywords": ["bleach", "chlorine", "disinfectant", "cure", "drink", "inject"],
        "verdict": "FALSE - DANGEROUS",
        "explanation": "Drinking or injecting bleach/disinfectants is extremely dangerous and can be fatal. They do NOT cure COVID-19.",
        "sources": ["CDC", "FDA", "WHO", "Poison Control Centers"],
        "confidence": 0.99
    },
    "flat_earth": {
        "patterns": [
            r"earth.*(flat|not.*round|not.*globe)",
            r"nasa.*(lie|fake|hoax)",
            r"globe.*(lie|fake|hoax)",
            r"flat.*earth.*society",
        ],
        "keywords": ["flat", "earth", "nasa", "globe", "lie", "hoax"],
        "verdict": "FALSE",
        "explanation": "The Earth is an oblate spheroid. This is confirmed by satellite imagery, physics, and thousands of years of scientific observation.",
        "sources": ["NASA", "ESA", "Every space agency", "Physics"],
        "confidence": 0.99
    },
    "moon_landing_fake": {
        "patterns": [
            r"moon.*landing.*(fake|hoax|studio|hollywood)",
            r"never.*went.*moon",
            r"apollo.*(fake|hoax|staged)",
        ],
        "keywords": ["moon", "landing", "fake", "hoax", "apollo", "nasa"],
        "verdict": "FALSE",
        "explanation": "The Apollo moon landings are one of the most well-documented events in history. Evidence includes moon rocks, retroreflectors still used today, and independent verification by other countries.",
        "sources": ["NASA", "Independent scientists", "Other space agencies"],
        "confidence": 0.98
    },
    "election_fraud_2020": {
        "patterns": [
            r"(2020|election).*(stolen|rigged|fraud)",
            r"(millions|mass).*fake.*ballot",
            r"dominion.*(switch|fraud|rigged)",
            r"dead.*people.*vot",
        ],
        "keywords": ["election", "stolen", "rigged", "fraud", "ballot", "dominion"],
        "verdict": "FALSE",
        "explanation": "The 2020 US election was extensively audited and verified. Over 60 court cases found no evidence of widespread fraud. Claims were rejected by Republican and Democratic officials alike.",
        "sources": ["CISA", "State election officials", "Courts", "Independent audits"],
        "confidence": 0.95
    },
    "chemtrails": {
        "patterns": [
            r"chemtrail",
            r"plane.*(spray|poison|chemical)",
            r"contrail.*(chemical|poison)",
            r"government.*spray.*population",
        ],
        "keywords": ["chemtrail", "contrail", "spray", "poison", "planes"],
        "verdict": "FALSE",
        "explanation": "Contrails are simply water vapor that condenses when hot jet exhaust meets cold air. There is no evidence of chemical spraying programs.",
        "sources": ["EPA", "NASA", "Atmospheric scientists"],
        "confidence": 0.97
    },
    "vaccine_autism": {
        "patterns": [
            r"vaccine.*(cause|link).*autism",
            r"mmr.*(cause|autism)",
            r"thimerosal.*(autism|mercury)",
        ],
        "keywords": ["vaccine", "autism", "mmr", "mercury", "thimerosal"],
        "verdict": "FALSE",
        "explanation": "Numerous large-scale studies have found NO link between vaccines and autism. The original study claiming this was retracted and its author lost his medical license for fraud.",
        "sources": ["CDC", "WHO", "Multiple meta-analyses", "Lancet retraction"],
        "confidence": 0.99
    },
    "covid_lab_leak_intentional": {
        "patterns": [
            r"covid.*(bioweapon|intentional|released.*purpose)",
            r"china.*(created|released).*covid.*purpose",
            r"planned.*pandemic",
        ],
        "keywords": ["bioweapon", "intentional", "planned", "released"],
        "verdict": "UNVERIFIED/DISPUTED",
        "explanation": "While the lab leak hypothesis is under investigation, claims that COVID was intentionally created or released as a bioweapon lack evidence.",
        "sources": ["Intelligence agencies", "Scientific investigations"],
        "confidence": 0.75
    },
    "ivermectin_covid": {
        "patterns": [
            r"ivermectin.*(cure|treat|prevent).*covid",
            r"horse.*(paste|dewormer).*covid",
        ],
        "keywords": ["ivermectin", "horse", "dewormer", "cure", "covid"],
        "verdict": "FALSE/UNPROVEN",
        "explanation": "Clinical trials have not shown ivermectin to be effective against COVID-19. The FDA warns against using veterinary ivermectin.",
        "sources": ["FDA", "WHO", "Clinical trials"],
        "confidence": 0.90
    },
    "climate_hoax": {
        "patterns": [
            r"climate.*change.*(hoax|fake|lie)",
            r"global.*warming.*(hoax|fake|scam)",
            r"scientists.*(lie|fake).*climate",
        ],
        "keywords": ["climate", "hoax", "fake", "lie", "scam", "warming"],
        "verdict": "FALSE",
        "explanation": "Climate change is supported by overwhelming scientific consensus (97%+ of climate scientists). Evidence includes temperature records, ice cores, and observable changes.",
        "sources": ["NASA", "NOAA", "IPCC", "97% of climate scientists"],
        "confidence": 0.98
    },
    "pizzagate": {
        "patterns": [
            r"pizzagate",
            r"comet.*ping.*pong.*(child|traffic|pedo)",
            r"pizza.*(code|child|traffic)",
        ],
        "keywords": ["pizzagate", "comet", "ping", "pong", "pizza", "code"],
        "verdict": "FALSE",
        "explanation": "Pizzagate is a debunked conspiracy theory. No evidence was ever found, and the claims were thoroughly investigated and dismissed.",
        "sources": ["FBI", "DC Police", "Multiple fact-checkers"],
        "confidence": 0.99
    },
    "qanon": {
        "patterns": [
            r"qanon",
            r"deep.*state.*cabal",
            r"storm.*is.*coming",
            r"great.*awakening",
        ],
        "keywords": ["qanon", "deep state", "cabal", "storm", "awakening"],
        "verdict": "FALSE",
        "explanation": "QAnon is a debunked far-right conspiracy theory. Its predictions have repeatedly failed to materialize.",
        "sources": ["FBI (designated as domestic terror threat)", "Multiple investigations"],
        "confidence": 0.95
    }
}

# Trusted fact-check sources and their credibility
TRUSTED_SOURCES = {
    "snopes.com": 0.9,
    "factcheck.org": 0.9,
    "politifact.com": 0.85,
    "reuters.com/fact-check": 0.95,
    "apnews.com/APFactCheck": 0.95,
    "bbc.com/news/reality_check": 0.9,
    "fullfact.org": 0.9,
    "who.int": 0.95,
    "cdc.gov": 0.95,
}


def check_known_misinformation(text: str) -> Dict[str, Any]:
    """
    Check text against known misinformation patterns
    
    Args:
        text: Text to check
        
    Returns:
        Match result with verdict and explanation
    """
    text_lower = text.lower()
    
    matches = []
    
    for topic_id, topic_info in KNOWN_MISINFORMATION.items():
        # Check regex patterns
        pattern_match = False
        for pattern in topic_info["patterns"]:
            if re.search(pattern, text_lower):
                pattern_match = True
                break
        
        # Check keyword density
        keyword_count = sum(1 for kw in topic_info["keywords"] if kw in text_lower)
        keyword_ratio = keyword_count / len(topic_info["keywords"])
        
        if pattern_match or keyword_ratio >= 0.4:
            matches.append({
                "topic": topic_id,
                "verdict": topic_info["verdict"],
                "explanation": topic_info["explanation"],
                "sources": topic_info["sources"],
                "confidence": topic_info["confidence"] * (0.9 if pattern_match else 0.7),
                "matched_by": "pattern" if pattern_match else "keywords"
            })
    
    if matches:
        # Return highest confidence match
        best_match = max(matches, key=lambda x: x["confidence"])
        return {
            "matched": True,
            "all_matches": matches,
            **best_match
        }
    
    return {
        "matched": False,
        "verdict": "UNVERIFIED",
        "explanation": "No known misinformation patterns detected. This doesn't mean the claim is true - it just hasn't matched our database.",
        "sources": [],
        "confidence": 0.0
    }


def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


async def search_google_factcheck(claim: str, api_key: Optional[str] = None) -> List[Dict]:
    """
    Search Google Fact Check Tools API
    
    Args:
        claim: The claim to search for
        api_key: Google API key (optional)
        
    Returns:
        List of fact-check results
    """
    if not api_key:
        # Try to get from environment
        import os
        api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    
    if not api_key:
        logger.warning("No Google Fact Check API key configured")
        return []
    
    results = []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={
                    "key": api_key,
                    "query": claim,
                    "languageCode": "en"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                for claim_data in data.get("claims", []):
                    for review in claim_data.get("claimReview", []):
                        publisher = review.get("publisher", {}).get("name", "Unknown")
                        results.append({
                            "source": publisher,
                            "url": review.get("url", ""),
                            "title": review.get("title", ""),
                            "rating": review.get("textualRating", ""),
                            "claim_reviewed": claim_data.get("text", ""),
                            "credibility": TRUSTED_SOURCES.get(
                                review.get("publisher", {}).get("site", "").lower(), 
                                0.7
                            )
                        })
            else:
                logger.warning(f"Fact check API returned {response.status_code}")
                
    except Exception as e:
        logger.error(f"Error calling Fact Check API: {e}")
    
    return results


async def verify_claim(claim: str, use_external_api: bool = False) -> Dict[str, Any]:
    """
    Comprehensive claim verification
    
    Args:
        claim: The claim/statement to verify
        use_external_api: Whether to use external fact-check APIs
        
    Returns:
        Verification result with verdict, sources, and confidence
    """
    logger.info(f"Verifying claim: {claim[:100]}...")
    
    # Step 1: Check local knowledge base
    local_check = check_known_misinformation(claim)
    
    # Step 2: Get external fact-checks if enabled
    external_checks = []
    if use_external_api:
        external_checks = await search_google_factcheck(claim)
    
    # Step 3: Analyze stance of fact-checks against claim
    stance_results = []
    if external_checks:
        for fc in external_checks[:5]:  # Limit to 5 sources
            try:
                combined_text = f"Claim: {claim}\nFact-check: {fc['rating']} - {fc.get('title', '')}"
                stance = predict_stance(combined_text, claim=claim)
                stance_results.append({
                    **fc,
                    "stance": stance["label"],
                    "stance_confidence": stance["confidence"]
                })
            except Exception as e:
                logger.error(f"Error analyzing stance: {e}")
    
    # Step 4: Calculate final verdict
    verdict, confidence, reasoning = calculate_verdict(local_check, stance_results)
    
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "local_match": local_check if local_check["matched"] else None,
        "fact_checks": stance_results,
        "sources_checked": len(stance_results) + (1 if local_check["matched"] else 0),
        "recommendation": get_recommendation(verdict, confidence)
    }


def calculate_verdict(local_check: Dict, stance_results: List[Dict]) -> tuple:
    """
    Calculate final verdict based on all evidence
    
    Returns:
        (verdict, confidence, reasoning)
    """
    # If local match with high confidence
    if local_check["matched"] and local_check["confidence"] >= 0.8:
        return (
            local_check["verdict"],
            local_check["confidence"],
            f"Matched known misinformation pattern: {local_check['topic']}. {local_check['explanation']}"
        )
    
    # If we have external fact-checks
    if stance_results:
        disagree_count = sum(1 for r in stance_results if r["stance"] == "disagree")
        agree_count = sum(1 for r in stance_results if r["stance"] == "agree")
        total = len(stance_results)
        
        # Check ratings from fact-checkers
        false_ratings = sum(1 for r in stance_results 
                          if any(word in r.get("rating", "").lower() 
                                for word in ["false", "fake", "misleading", "incorrect", "wrong", "debunked"]))
        
        if false_ratings >= total * 0.5 or disagree_count >= total * 0.6:
            confidence = max(0.7, false_ratings / total)
            return (
                "LIKELY FALSE",
                confidence,
                f"{false_ratings}/{total} fact-checkers rated this as false or misleading"
            )
        elif agree_count >= total * 0.6:
            confidence = agree_count / total * 0.9
            return (
                "LIKELY TRUE",
                confidence,
                f"{agree_count}/{total} fact-checkers support this claim"
            )
        else:
            return (
                "DISPUTED",
                0.5,
                "Fact-checkers have mixed opinions on this claim"
            )
    
    # Weak local match
    if local_check["matched"]:
        return (
            local_check["verdict"],
            local_check["confidence"],
            local_check["explanation"]
        )
    
    # No information
    return (
        "UNVERIFIED",
        0.0,
        "Unable to verify this claim. No matching fact-checks found."
    )


def get_recommendation(verdict: str, confidence: float) -> str:
    """Get user-friendly recommendation based on verdict"""
    if "FALSE" in verdict or "FAKE" in verdict:
        if confidence >= 0.9:
            return "⚠️ This claim has been thoroughly debunked. Do not share."
        elif confidence >= 0.7:
            return "⚠️ This claim is likely false. Verify before sharing."
        else:
            return "🔍 This claim matches misinformation patterns. Research further."
    elif "TRUE" in verdict:
        if confidence >= 0.8:
            return "✅ This claim appears to be accurate."
        else:
            return "✅ This claim is likely true, but verify important details."
    elif verdict == "DISPUTED":
        return "⚖️ This claim is disputed by experts. Read multiple sources."
    else:
        return "❓ Unable to verify. Check reputable news sources."


def quick_claim_check(text: str) -> Dict[str, Any]:
    """
    Synchronous quick check against known misinformation
    Use this for real-time checks without external API calls
    
    Args:
        text: Text to check
        
    Returns:
        Quick verification result
    """
    from app.services.model_loader import predict_misinfo, predict_sentiment
    
    result = check_known_misinformation(text)
    
    # Even if no known pattern matched, use the ML model to analyze
    misinfo_result = None
    sentiment_result = None
    
    try:
        misinfo_result = predict_misinfo(text)
        sentiment_result = predict_sentiment(text)
    except Exception as e:
        logger.error(f"Error running ML models: {e}")
    
    # If known misinformation matched, return it
    if result["matched"]:
        return {
            "claim": text[:200] + "..." if len(text) > 200 else text,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "matched": result["matched"],
            "explanation": result["explanation"],
            "sources": result.get("sources", []),
            "recommendation": get_recommendation(result["verdict"], result["confidence"]),
            "ml_analysis": {
                "misinfo_risk": misinfo_result["label"] if misinfo_result else None,
                "misinfo_confidence": misinfo_result["confidence"] if misinfo_result else None,
                "sentiment": sentiment_result["label"] if sentiment_result else None,
            } if misinfo_result else None
        }
    
    # No pattern matched - use ML model analysis
    if misinfo_result:
        ml_verdict = "UNVERIFIED"
        ml_confidence = 0.0
        ml_explanation = "No known misinformation patterns detected."
        
        misinfo_label = misinfo_result["label"].lower()
        misinfo_score = misinfo_result["confidence"]
        
        if misinfo_label in ["misinformation", "misinfo", "fake", "false"]:
            if misinfo_score >= 0.8:
                ml_verdict = "LIKELY MISLEADING"
                ml_confidence = misinfo_score * 0.85  # Slightly reduced since ML-based
                ml_explanation = f"ML analysis indicates high misinformation risk ({round(misinfo_score*100)}%). This content may contain misleading claims."
            elif misinfo_score >= 0.6:
                ml_verdict = "POSSIBLY MISLEADING"
                ml_confidence = misinfo_score * 0.7
                ml_explanation = f"ML analysis detected moderate misinformation indicators ({round(misinfo_score*100)}%). Exercise caution."
            else:
                ml_verdict = "LOW RISK"
                ml_confidence = 1 - misinfo_score
                ml_explanation = f"ML analysis shows low misinformation risk ({round(misinfo_score*100)}%)."
        else:
            # Not flagged as misinfo
            ml_verdict = "NO MISINFO DETECTED"
            ml_confidence = 1 - misinfo_score if misinfo_score < 0.5 else 0.5
            ml_explanation = f"ML analysis did not flag significant misinformation indicators. Content appears factual based on language patterns."
        
        return {
            "claim": text[:200] + "..." if len(text) > 200 else text,
            "verdict": ml_verdict,
            "confidence": ml_confidence,
            "matched": False,
            "explanation": ml_explanation,
            "sources": ["ML-based analysis (DeBERTa misinfo model)"],
            "recommendation": get_recommendation(ml_verdict, ml_confidence),
            "ml_analysis": {
                "misinfo_risk": misinfo_result["label"],
                "misinfo_confidence": misinfo_result["confidence"],
                "sentiment": sentiment_result["label"] if sentiment_result else None,
                "sentiment_confidence": sentiment_result.get("score", sentiment_result.get("confidence")) if sentiment_result else None,
            }
        }
    
    # Fallback if ML models failed
    return {
        "claim": text[:200] + "..." if len(text) > 200 else text,
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "matched": result["matched"],
        "explanation": result["explanation"],
        "sources": result.get("sources", []),
        "recommendation": get_recommendation(result["verdict"], result["confidence"])
    }
