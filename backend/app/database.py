"""
Database setup and models for logging analysis results
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from app.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AnalysisLog(Base):
    """Log table for all analysis requests"""
    __tablename__ = "analysis_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Input
    text_hash = Column(String(64), index=True)  # SHA256 hash (not raw text for privacy)
    text_length = Column(Integer)
    source_platform = Column(String(50))  # twitter, instagram, youtube
    
    # Sentiment
    sentiment_label = Column(String(20))
    sentiment_score = Column(Float)
    
    # Topics/Narrative
    topic_id = Column(Integer)
    topic_keywords = Column(String(500))
    
    # Framing
    frame_label = Column(String(50))
    frame_confidence = Column(Float)
    
    # Misinformation
    misinfo_risk = Column(Float)
    misinfo_triggers = Column(Text)
    
    # Baseline Comparison
    narrative_distance = Column(Float)
    baseline_event = Column(String(200))
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Integer)
    anonymized = Column(Boolean, default=True)


class CrawlCache(Base):
    """Cache for crawled news results to avoid re-crawling"""
    __tablename__ = "crawl_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String(64), index=True)  # SHA256 of query
    query = Column(String(1000))
    article_title = Column(String(500))
    article_url = Column(String(2000))
    article_source = Column(String(200))
    article_domain = Column(String(200))
    snippet = Column(Text)
    crawl_source = Column(String(50))  # google_news, gdelt, newsapi
    credibility_score = Column(Float, nullable=True)
    verdict = Column(String(50), nullable=True)
    cached_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # Cache TTL


class BaselineEvent(Base):
    """Cached baseline events from GDELT/FEVER"""
    __tablename__ = "baseline_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(100), unique=True)
    source = Column(String(20))  # gdelt, fever
    title = Column(String(500))
    description = Column(Text)
    embedding = Column(Text)  # JSON serialized embedding
    event_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
