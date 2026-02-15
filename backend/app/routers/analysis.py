"""
Main analysis endpoint - unified pipeline
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import time
import hashlib
from datetime import datetime

from app.models.schemas import (
    AnalysisRequest, 
    AnalysisResponse,
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    VideoAnalysisRequest,
    VideoAnalysisResponse,
    SentimentResult,
    StanceResult,
    TopicResult,
    FramingResult,
    MisinfoResult,
    BaselineComparison,
    ExplainabilityOutput,
    OCRResult,
    ImageContentResult,
    # New schemas for thread/claim analysis
    ThreadAnalysisRequest,
    ThreadAnalysisResponse,
    QuoteTweetRequest,
    QuoteTweetResponse,
    ClaimVerificationRequest,
    ClaimVerificationResponse,
    FullThreadRequest,
    FullThreadResponse
)
from app.database import get_db, AnalysisLog
from app.services.text_cleaner import clean_text, anonymize_text
from app.services.sentiment import analyze_sentiment
from app.services.stance import analyze_stance
from app.services.topics import extract_topics
from app.services.framing import detect_framing
from app.services.misinfo import detect_misinformation
from app.services.baseline import compare_baseline
from app.services.explainer import generate_explanation
from app.services.image_analysis import analyze_image
from app.services.news_crawler import NewsCrawler, TRUSTED_SOURCES, KNOWN_UNRELIABLE_SOURCES, SATIRE_SOURCES, extract_search_keywords
from app.services.credibility_analyzer import analyze_credibility, _text_similarity
from app.models.schemas import VerifiedSource
from app.config import settings

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Unified analysis pipeline:
    1. Text cleaning & anonymization
    2. Sentiment analysis
    3. Topic/Narrative extraction
    4. Framing/Propaganda detection
    5. Misinformation detection
    6. Baseline comparison (GDELT/FEVER)
    7. Explainability generation
    """
    start_time = time.time()
    
    try:
        # Step 1: Clean and optionally anonymize text
        cleaned_text = clean_text(request.text)
        if request.anonymize:
            cleaned_text = anonymize_text(cleaned_text)
        
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            raise HTTPException(status_code=400, detail="Text too short after cleaning")
        
        # Step 2: Run all analysis modules (can be parallelized)
        sentiment = analyze_sentiment(cleaned_text)
        stance = analyze_stance(cleaned_text)  # Custom DeBERTa (93.15% F1)
        topics = extract_topics(cleaned_text)
        framing = detect_framing(cleaned_text)
        misinfo = detect_misinformation(cleaned_text)
        
        # Step 3: Baseline comparison (research novelty)
        baseline = BaselineComparison(
            narrative_distance=0.0,
            closest_event=None,
            event_source=None,
            deviation_type=None
        )
        if request.include_baseline:
            baseline = compare_baseline(cleaned_text, topics)
        
        # Step 3.5: Real-time source verification via crawler
        try:
            crawler = NewsCrawler()
            articles = await crawler.search_news(
                query=cleaned_text,       # crawlers extract keywords for API queries
                max_results=8,
                enrich_text=False,
                timespan="7d",
                original_text=cleaned_text  # full text used for semantic scoring
            )
            await crawler.close()
            
            if articles:
                cred_report = analyze_credibility(cleaned_text, articles)
                
                # Attach verified sources to misinfo result
                # Only include articles that are actually relevant to the claim
                search_kw = extract_search_keywords(cleaned_text, max_keywords=6).lower().split()
                verified = []
                for a in articles:
                    # Check if article title shares key terms with the post
                    title_lower = a.title.lower()
                    title_sim = _text_similarity(cleaned_text, a.title + " " + a.snippet)
                    kw_hits = sum(1 for kw in search_kw if kw.lower() in title_lower)

                    # Skip articles that don't match the topic at all
                    if title_sim < 0.04 and kw_hits < 2:
                        continue

                    domain = a.source_domain.lower()
                    if any(t in domain for t in TRUSTED_SOURCES):
                        trust = "trusted"
                    elif any(u in domain for u in KNOWN_UNRELIABLE_SOURCES):
                        trust = "unreliable"
                    elif any(s in domain for s in SATIRE_SOURCES):
                        trust = "satire"
                    else:
                        trust = "unknown"
                    verified.append(VerifiedSource(
                        title=a.title[:150],
                        url=a.url,
                        source=a.source,
                        domain=a.source_domain,
                        trust_level=trust
                    ))
                    if len(verified) >= 6:
                        break
                
                misinfo.verified_sources = verified
                misinfo.verification_verdict = cred_report.get("verdict", None)
                misinfo.credibility_score = cred_report.get("credibility_score", None)
                
                # ── Reconcile misinfo risk with crawler evidence ──
                # The DeBERTa misinfo model was trained on news articles, not
                # social-media posts.  When the crawler finds real coverage of
                # the event from trusted sources, we adjust the risk score
                # downward to reflect the real-world evidence.
                cred_score = cred_report.get("credibility_score", 0.0)
                verdict = cred_report.get("verdict", "")
                trusted_found = sum(1 for v in verified if v.trust_level == "trusted")
                matching = cred_report.get("cross_reference", {}).get("matching_headlines", 0)
                
                if verdict in ("LIKELY_AUTHENTIC", "POSSIBLY_AUTHENTIC"):
                    # Crawler says the event is confirmed → lower misinfo risk
                    adjusted_risk = misinfo.risk_score * (1 - cred_score)
                    # Floor: don't go below 0.05 (some risk always remains)
                    misinfo.risk_score = round(max(adjusted_risk, 0.05), 4)
                    if adjusted_risk < 0.4:
                        misinfo.risk_level = "low"
                    elif adjusted_risk < 0.65:
                        misinfo.risk_level = "medium"
                    logger.info(
                        f"Crawler override: risk {misinfo.risk_score} "
                        f"(cred={cred_score:.2f}, verdict={verdict}, "
                        f"trusted={trusted_found}, matching={matching})"
                    )
                elif verdict == "UNVERIFIED" and trusted_found >= 1:
                    # Trusted sources found but not fully conclusive
                    adjusted_risk = misinfo.risk_score * 0.5
                    misinfo.risk_score = round(max(adjusted_risk, 0.1), 4)
                    if adjusted_risk < 0.65:
                        misinfo.risk_level = "medium"
                    logger.info(f"Crawler partial override (trusted): risk {misinfo.risk_score}")
                elif verdict == "UNVERIFIED" and matching >= 1 and len(verified) >= 1:
                    # At least one article headline matched the claim —
                    # even if the source isn't in our "trusted" list, the
                    # event is clearly being reported on.  Reduce risk.
                    adjusted_risk = misinfo.risk_score * 0.65
                    misinfo.risk_score = round(max(adjusted_risk, 0.15), 4)
                    if adjusted_risk < 0.65:
                        misinfo.risk_level = "medium"
                    logger.info(
                        f"Crawler partial override (matched): risk {misinfo.risk_score} "
                        f"(matching={matching}, verified={len(verified)})"
                    )
        except Exception as e:
            logger.warning(f"Crawler verification skipped: {e}")
        
        # Step 5: Generate explanation
        explanation = generate_explanation(
            text=cleaned_text,
            sentiment=sentiment,
            topics=topics,
            framing=framing,
            misinfo=misinfo,
            baseline=baseline
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Step 6: Log to database (optional)
        if settings.LOG_ANALYSIS:
            log_entry = AnalysisLog(
                text_hash=hashlib.sha256(request.text.encode()).hexdigest(),
                text_length=len(request.text),
                source_platform=request.source,
                sentiment_label=sentiment.label,
                sentiment_score=sentiment.score,
                topic_id=topics.topic_id,
                topic_keywords=",".join(topics.keywords),
                frame_label=framing.frame,
                frame_confidence=framing.confidence,
                misinfo_risk=misinfo.risk_score,
                misinfo_triggers=",".join(misinfo.triggers),
                narrative_distance=baseline.narrative_distance,
                baseline_event=baseline.closest_event,
                processing_time_ms=processing_time,
                anonymized=request.anonymize
            )
            db.add(log_entry)
            db.commit()
        
        return AnalysisResponse(
            sentiment=sentiment,
            stance=stance,
            topics=topics,
            framing=framing,
            misinformation=misinfo,
            baseline=baseline,
            explanation=explanation,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
            anonymized=request.anonymize
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(
    request: ImageAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Image analysis pipeline:
    1. Fetch/decode image
    2. OCR - Extract text from image
    3. CLIP - Analyze image content & detect manipulation
    4. Run text analysis on extracted text
    5. Combine results for overall assessment
    """
    start_time = time.time()
    
    try:
        # Determine image source
        image_source = request.image_url or request.image_base64
        if not image_source:
            raise HTTPException(status_code=400, detail="Must provide image_url or image_base64")
        
        # Step 1: Analyze image (OCR + CLIP)
        img_result = analyze_image(
            image_source=image_source,
            extract_text=request.extract_text,
            analyze_content=request.analyze_content
        )
        
        if not img_result['success']:
            raise HTTPException(status_code=400, detail=img_result.get('error', 'Image analysis failed'))
        
        # Step 2: Combine extracted text with provided caption
        combined_text = ""
        if request.text:
            combined_text = request.text
        if img_result.get('combined_text'):
            combined_text = f"{combined_text} {img_result['combined_text']}".strip()
        
        # Step 3: Run text analysis if we have text
        text_analysis = None
        risk_factors = []
        overall_risk = "low"
        
        if combined_text and len(combined_text) >= 10:
            # Clean and analyze text
            text_start_time = time.time()
            cleaned_text = clean_text(combined_text)
            
            sentiment = analyze_sentiment(cleaned_text)
            topics = extract_topics(cleaned_text)
            framing = detect_framing(cleaned_text)
            misinfo = detect_misinformation(cleaned_text)
            stance = analyze_stance(cleaned_text)  # Add stance
            baseline = compare_baseline(cleaned_text, topics)
            explanation = generate_explanation(
                text=cleaned_text,
                sentiment=sentiment,
                topics=topics,
                framing=framing,
                misinfo=misinfo,
                baseline=baseline
            )
            
            text_processing_time = int((time.time() - text_start_time) * 1000)
            
            text_analysis = AnalysisResponse(
                sentiment=sentiment,
                stance=stance,
                topics=topics,
                framing=framing,
                misinformation=misinfo,
                baseline=baseline,
                explanation=explanation,
                processing_time_ms=text_processing_time,
                timestamp=datetime.utcnow(),
                anonymized=False
            )
            
            # Check risk factors from text
            if misinfo.risk_level == "high":
                risk_factors.append("High misinformation risk in text")
                overall_risk = "high"
            elif misinfo.risk_level == "medium" and overall_risk != "high":
                risk_factors.append("Medium misinformation risk in text")
                overall_risk = "medium"
        
        # Step 4: Check image-specific risk factors
        if img_result.get('is_manipulated'):
            risk_factors.append("Image may be digitally manipulated")
            overall_risk = "high" if overall_risk != "high" else overall_risk
        
        content = img_result.get('content')
        if content:
            manipulation_score = content.get('manipulation_score', 0)
            if manipulation_score > 0.5:
                risk_factors.append(f"High manipulation score: {manipulation_score:.1%}")
                overall_risk = "high"
            elif manipulation_score > 0.3:
                risk_factors.append(f"Moderate manipulation score: {manipulation_score:.1%}")
                if overall_risk == "low":
                    overall_risk = "medium"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Build response
        ocr_result = None
        if img_result.get('ocr'):
            ocr_data = img_result['ocr']
            ocr_result = OCRResult(
                text=ocr_data['text'],
                segments=ocr_data['segments'],
                confidence=ocr_data['confidence']
            )
        
        content_result = None
        if content:
            content_result = ImageContentResult(
                labels=content['labels'],
                description=content['description'],
                is_manipulated=content['is_manipulated'],
                manipulation_score=content.get('manipulation_score'),
                image_type=content['image_type']
            )
        
        return ImageAnalysisResponse(
            success=True,
            image_size=img_result.get('image_size'),
            ocr=ocr_result,
            extracted_text=img_result.get('combined_text', ''),
            content=content_result,
            image_type=img_result.get('image_type', 'unknown'),
            is_manipulated=img_result.get('is_manipulated', False),
            text_analysis=text_analysis,
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return ImageAnalysisResponse(
            success=False,
            processing_time_ms=int((time.time() - start_time) * 1000),
            timestamp=datetime.utcnow(),
            error=str(e)
        )


@router.post("/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video_endpoint(
    request: VideoAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze video or GIF content:
    1. Extract frames at intervals
    2. OCR text from frames  
    3. CLIP content analysis
    4. Detect video-specific misinfo signals
    5. Combine with text analysis
    """
    from app.services.video_analysis import analyze_video_or_gif, get_video_misinfo_boost
    
    start_time = time.time()
    
    try:
        # Get media source
        media_source = request.media_url or request.media_base64
        if not media_source:
            raise HTTPException(status_code=400, detail="Either media_url or media_base64 must be provided")
        
        # Analyze video/GIF
        video_result = analyze_video_or_gif(
            media_source=media_source,
            max_frames=request.max_frames
        )
        
        if video_result['frame_count'] == 0:
            return VideoAnalysisResponse(
                success=False,
                processing_time_ms=int((time.time() - start_time) * 1000),
                timestamp=datetime.utcnow(),
                error="Could not extract frames from media"
            )
        
        # Combine text: caption + extracted text from frames
        combined_text = ""
        if request.text:
            combined_text = request.text + " "
        combined_text += video_result['combined_text']
        
        # Run text analysis if we have text
        text_analysis = None
        risk_factors = list(video_result['misinfo_signals'])
        overall_risk = "low"
        video_risk_boost = 0.0
        
        if combined_text.strip():
            # Clean text
            text_start_time = time.time()
            cleaned_text = clean_text(combined_text)
            
            # Run analysis pipeline
            sentiment = analyze_sentiment(cleaned_text)
            topics = extract_topics(cleaned_text)
            framing = detect_framing(cleaned_text)
            misinfo = detect_misinformation(cleaned_text)
            
            baseline = BaselineComparison(
                narrative_distance=0.0,
                closest_event=None,
                event_source=None,
                deviation_type=None
            )
            
            # Misinfo detection done - model output used directly
            
            # Get video-specific risk boost
            video_boost, video_triggers = get_video_misinfo_boost(video_result)
            video_risk_boost = video_boost
            risk_factors.extend(video_triggers)
            
            # Apply video boost to misinfo score
            final_risk_score = min(misinfo.risk_score + video_boost, 1.0)
            
            # Determine final risk level
            if final_risk_score >= 0.65:
                final_risk_level = "high"
            elif final_risk_score >= 0.4:
                final_risk_level = "medium"
            else:
                final_risk_level = "low"
            
            # Create updated misinfo result
            final_misinfo = MisinfoResult(
                risk_score=round(final_risk_score, 4),
                risk_level=final_risk_level,
                triggers=list(set(misinfo.triggers + video_triggers))[:8],
                claim_type=misinfo.claim_type
            )
            
            # Generate explanation
            explanation = generate_explanation(
                text=cleaned_text,
                sentiment=sentiment,
                topics=topics,
                framing=framing,
                misinfo=final_misinfo,
                baseline=baseline
            )
            
            stance = analyze_stance(cleaned_text)  # Add stance
            
            text_processing_time = int((time.time() - text_start_time) * 1000)
            
            text_analysis = AnalysisResponse(
                sentiment=sentiment,
                stance=stance,
                topics=topics,
                framing=framing,
                misinformation=final_misinfo,
                baseline=baseline,
                explanation=explanation,
                processing_time_ms=text_processing_time,
                timestamp=datetime.utcnow(),
                anonymized=False  # No anonymization applied for video text extraction
            )
            
            overall_risk = final_risk_level
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return VideoAnalysisResponse(
            success=True,
            media_type=video_result['media_type'],
            frame_count=video_result['frame_count'],
            duration_seconds=video_result.get('duration_seconds', 0.0),
            ocr_results=video_result['ocr_results'],
            extracted_text=video_result['combined_text'],
            frame_descriptions=video_result['frame_descriptions'],
            video_signals=video_result['misinfo_signals'],
            text_analysis=text_analysis,
            overall_risk=overall_risk,
            risk_factors=list(set(risk_factors)),
            video_risk_boost=video_risk_boost,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return VideoAnalysisResponse(
            success=False,
            processing_time_ms=int((time.time() - start_time) * 1000),
            timestamp=datetime.utcnow(),
            error=str(e)
        )


# ============================================================
# THREAD & CLAIM VERIFICATION ENDPOINTS
# ============================================================

@router.post("/analyze-thread", response_model=ThreadAnalysisResponse)
async def analyze_thread(request: ThreadAnalysisRequest):
    """
    Analyze stance of replies against an original post.
    
    Useful for understanding:
    - How people react to a post
    - Whether a claim is being disputed or supported
    - Community consensus on controversial topics
    
    Returns stance breakdown and credibility signals.
    """
    from app.services.stance import analyze_reply_chain
    
    try:
        result = analyze_reply_chain(
            original_text=request.original_text,
            replies=request.replies
        )
        
        return ThreadAnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thread analysis failed: {str(e)}")


@router.post("/analyze-quote", response_model=QuoteTweetResponse)
async def analyze_quote_tweet(request: QuoteTweetRequest):
    """
    Analyze the relationship between a quoted tweet and its commentary.
    
    Determines if the user is:
    - Amplifying (agreeing/sharing)
    - Critiquing (disagreeing/debunking)
    - Commenting (adding context)
    - Tangential (off-topic, ratio attempt)
    
    Useful for understanding content spread patterns.
    """
    from app.services.stance import analyze_quote_tweet
    
    try:
        result = analyze_quote_tweet(
            quoted_text=request.quoted_text,
            commentary=request.commentary
        )
        
        return QuoteTweetResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quote analysis failed: {str(e)}")


@router.post("/verify-claim", response_model=ClaimVerificationResponse)
async def verify_claim(request: ClaimVerificationRequest):
    """
    Verify a claim against known misinformation patterns and fact-check sources.
    
    Checks against:
    - Local database of debunked claims (5G, vaccines, election, etc.)
    - Google Fact Check API (if use_external_api=True)
    
    Returns verdict: TRUE, FALSE, DISPUTED, or UNVERIFIED
    """
    from app.services.claim_verification import verify_claim as verify
    
    try:
        result = await verify(
            claim=request.claim,
            use_external_api=request.use_external_api
        )
        
        return ClaimVerificationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claim verification failed: {str(e)}")


@router.post("/verify-claim-quick")
async def verify_claim_quick(request: ClaimVerificationRequest):
    """
    Quick synchronous claim verification (no external API calls).
    
    Faster than /verify-claim but only checks local misinformation database.
    Best for real-time UI feedback.
    """
    from app.services.claim_verification import quick_claim_check
    
    try:
        result = quick_claim_check(request.claim)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick verification failed: {str(e)}")


@router.post("/analyze-full-thread", response_model=FullThreadResponse)
async def analyze_full_thread(request: FullThreadRequest):
    """
    Analyze a complete thread for credibility.
    
    Takes an ordered list of posts (first is original, rest are replies)
    and determines:
    - Overall thread credibility
    - How the community reacts to the original claim
    - Whether the claim is being disputed or supported
    
    Useful for understanding viral threads and misinformation spread.
    """
    from app.services.stance import analyze_thread_credibility
    
    try:
        result = analyze_thread_credibility(request.posts)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return FullThreadResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thread credibility analysis failed: {str(e)}")
