"""
News Crawler & Real-Time Credibility Verification Router
Endpoints for searching news, verifying claims, and detecting misinformation in real-time.
"""
from fastapi import APIRouter, HTTPException
import time
import logging
from datetime import datetime

from app.models.schemas import (
    CrawlerSearchRequest,
    CrawlerSearchResponse,
    CrawledArticleSchema,
    CredibilityReportSchema,
    SourceAnalysisSchema,
    ConsistencyAnalysisSchema,
    SensationalismSchema,
    CrossReferenceSchema,
    QuickCheckRequest,
    QuickCheckResponse,
)
from app.services.news_crawler import NewsCrawler
from app.services.credibility_analyzer import analyze_credibility

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared crawler instance (created per-request, cleaned up after)


@router.post("/crawler/search", response_model=CrawlerSearchResponse)
async def crawler_search(request: CrawlerSearchRequest):
    """
    Search multiple news sources for articles related to a query/claim
    and optionally run credibility analysis.
    
    Sources searched:
    - Google News RSS (free, no key)
    - GDELT Project API (free, global coverage)
    - NewsAPI (optional, requires NEWSAPI_KEY env var)
    
    Pipeline:
    1. Parallel search across all sources
    2. Deduplicate results
    3. Score and rank by relevance
    4. Optionally enrich top articles with full text (web scraping)
    5. Run credibility analysis (cross-reference, source trust, AI model)
    """
    start_time = time.time()
    crawler = NewsCrawler()
    
    try:
        # Search all sources
        articles = await crawler.search_news(
            query=request.query,
            max_results=request.max_results,
            enrich_text=request.enrich_text,
            max_enrich=request.max_enrich,
            timespan=request.timespan,
            original_text=request.query  # use query as full text for semantic scoring
        )
        
        # Convert to response schema
        article_schemas = [
            CrawledArticleSchema(
                title=a.title,
                url=a.url,
                source=a.source,
                source_domain=a.source_domain,
                published_date=a.published_date,
                snippet=a.snippet[:300],
                has_full_text=bool(a.full_text),
                crawl_source=a.crawl_source,
            )
            for a in articles
        ]
        
        # Run credibility analysis if requested
        credibility_report = None
        if request.analyze_credibility and articles:
            raw_report = analyze_credibility(request.query, articles)
            credibility_report = CredibilityReportSchema(
                claim=raw_report["claim"],
                credibility_score=raw_report["credibility_score"],
                verdict=raw_report["verdict"],
                verdict_explanation=raw_report["verdict_explanation"],
                confidence=raw_report["confidence"],
                risk_level=raw_report["risk_level"],
                source_analysis=SourceAnalysisSchema(**raw_report["source_analysis"]),
                consistency_analysis=ConsistencyAnalysisSchema(**raw_report["consistency_analysis"]),
                sensationalism=SensationalismSchema(**raw_report["sensationalism"]),
                model_prediction=raw_report["model_prediction"],
                cross_reference=CrossReferenceSchema(**raw_report["cross_reference"]),
                supporting_articles=raw_report["supporting_articles"],
                contradicting_articles=raw_report["contradicting_articles"],
                articles_analyzed=raw_report["articles_analyzed"],
                timestamp=raw_report["timestamp"],
            )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return CrawlerSearchResponse(
            success=True,
            query=request.query,
            articles=article_schemas,
            total_articles=len(article_schemas),
            credibility_report=credibility_report,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Crawler search error: {e}")
        processing_time = int((time.time() - start_time) * 1000)
        return CrawlerSearchResponse(
            success=False,
            query=request.query,
            articles=[],
            total_articles=0,
            credibility_report=None,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
            error=str(e)
        )
    finally:
        await crawler.close()


@router.post("/crawler/quick-check", response_model=QuickCheckResponse)
async def quick_check(request: QuickCheckRequest):
    """
    Quick claim verification — single endpoint that:
    1. Searches news for the claim
    2. Runs credibility analysis
    3. Returns a simple verdict
    
    Ideal for browser extension / real-time use.
    """
    start_time = time.time()
    crawler = NewsCrawler()
    
    try:
        # Fast search with fewer results
        articles = await crawler.search_news(
            query=request.claim,
            max_results=10,
            enrich_text=True,
            max_enrich=3,
            timespan="7d",
            original_text=request.claim
        )
        
        # Run credibility analysis
        report = analyze_credibility(request.claim, articles)
        
        # Extract top sources for the response
        top_sources = []
        for article in articles[:5]:
            top_sources.append({
                "title": article.title[:100],
                "source": article.source,
                "url": article.url
            })
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return QuickCheckResponse(
            claim=request.claim,
            verdict=report["verdict"],
            confidence=report["confidence"],
            risk_level=report["risk_level"],
            explanation=report["verdict_explanation"],
            credibility_score=report["credibility_score"],
            articles_found=len(articles),
            trusted_sources=report["source_analysis"]["trusted_sources"],
            top_sources=top_sources,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Quick check error: {e}")
        processing_time = int((time.time() - start_time) * 1000)
        return QuickCheckResponse(
            claim=request.claim,
            verdict="ERROR",
            confidence=0.0,
            risk_level="unknown",
            explanation=f"Verification failed: {str(e)}",
            credibility_score=0.0,
            articles_found=0,
            trusted_sources=0,
            top_sources=[],
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )
    finally:
        await crawler.close()


@router.get("/crawler/sources")
async def list_sources():
    """
    List all known source registries (trusted, unreliable, satire)
    and their trust levels.
    """
    from app.services.news_crawler import TRUSTED_SOURCES, KNOWN_UNRELIABLE_SOURCES, SATIRE_SOURCES
    
    return {
        "trusted_sources": sorted(list(TRUSTED_SOURCES)),
        "unreliable_sources": sorted(list(KNOWN_UNRELIABLE_SOURCES)),
        "satire_sources": sorted(list(SATIRE_SOURCES)),
        "total_registered": len(TRUSTED_SOURCES) + len(KNOWN_UNRELIABLE_SOURCES) + len(SATIRE_SOURCES)
    }


@router.get("/crawler/check-source/{domain}")
async def check_source(domain: str):
    """
    Check the trust level of a specific news source domain.
    Example: /api/crawler/check-source/reuters.com
    """
    crawler = NewsCrawler()
    try:
        info = crawler.get_source_info(domain)
        return {
            "domain": domain,
            **info
        }
    finally:
        await crawler.close()
