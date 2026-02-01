"""
Configuration settings for the OSINT Monitoring Backend
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "OSINT Monitoring API"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./osint_logs.db"
    
    # Redis (optional caching)
    REDIS_URL: Optional[str] = None
    
    # GPU/CUDA Settings
    USE_GPU: bool = True
    CUDA_VISIBLE_DEVICES: str = "0"
    
    # Model Settings
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Thresholds
    MISINFO_THRESHOLD: float = 0.7
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Ethical Settings
    ANONYMIZE_TEXT: bool = True
    LOG_ANALYSIS: bool = True
    
    # GDELT API (for baseline comparison)
    GDELT_API_URL: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    class Config:
        env_file = ".env"


settings = Settings()
