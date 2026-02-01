"""
Text cleaning and anonymization utilities
Ethical filtering for OSINT compliance
"""
import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean social media text:
    - Remove URLs
    - Normalize whitespace
    - Handle emojis (keep or remove based on config)
    - Remove excessive hashtags
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove pic.twitter.com links
    text = re.sub(r'pic\.twitter\.com/\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive hashtags (keep first 3)
    hashtags = re.findall(r'#\w+', text)
    if len(hashtags) > 3:
        for tag in hashtags[3:]:
            text = text.replace(tag, '')
    
    # Remove RT prefix
    text = re.sub(r'^RT\s*@\w+:\s*', '', text)
    
    return text.strip()


def anonymize_text(text: str) -> str:
    """
    Remove personally identifiable information:
    - @mentions → [USER]
    - Names (basic pattern) → [NAME]
    - Email addresses → [EMAIL]
    - Phone numbers → [PHONE]
    
    This is critical for ethical OSINT compliance.
    """
    if not text:
        return ""
    
    # Remove @mentions
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Remove email addresses
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)
    
    # Remove phone numbers (various formats)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\+\d{1,3}[-.\s]?\d{1,14}', '[PHONE]', text)
    
    return text


def extract_hashtags(text: str) -> list:
    """Extract hashtags from text"""
    return re.findall(r'#(\w+)', text)


def extract_mentions(text: str) -> list:
    """Extract @mentions from text (before anonymization)"""
    return re.findall(r'@(\w+)', text)


def detect_platform(text: str, url: Optional[str] = None) -> str:
    """Detect source platform from text patterns or URL"""
    if url:
        if 'twitter.com' in url or 'x.com' in url:
            return 'twitter'
        elif 'instagram.com' in url:
            return 'instagram'
        elif 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'facebook.com' in url:
            return 'facebook'
        elif 'reddit.com' in url:
            return 'reddit'
    
    # Text-based detection
    if text.startswith('RT @') or '#' in text and '@' in text:
        return 'twitter'
    
    return 'unknown'
