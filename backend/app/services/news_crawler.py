"""
Real-Time News Crawler for Misinformation Detection
Searches multiple news sources, crawls articles, and cross-references
them to determine authenticity in real-time.

Sources:
  - Google News RSS (no API key needed)
  - GDELT Project API (free, real-time global news)
  - NewsAPI.org (optional, requires key)
  - Direct web scraping (fallback)
"""

import logging
import asyncio
import re
import hashlib
import json
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urlparse
from dataclasses import dataclass, field, asdict

import httpx
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

# Lazy-loaded SBERT embedder reference
_embedder = None


def _get_embedder():
    """Get the SBERT embedder from the model loader (lazy, shared instance)."""
    global _embedder
    if _embedder is None:
        try:
            from app.services.model_loader import get_model
            _embedder = get_model("embedder")
        except Exception:
            _embedder = None
    return _embedder


# ============================================================
# KEYWORD EXTRACTION — Extract short search queries from text
# ============================================================

# Common stop words to filter out
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "not", "no", "nor", "but", "and", "or", "so", "if", "then",
    "than", "too", "very", "just", "about", "above", "after", "again",
    "all", "also", "am", "any", "because", "before", "between", "both",
    "by", "come", "each", "few", "for", "from", "get", "got", "here",
    "in", "into", "of", "off", "on", "once", "only", "other", "out",
    "over", "own", "per", "same", "some", "such", "to", "under", "until",
    "up", "with", "don", "t", "s", "re", "ve", "ll", "d", "m",
    "most", "more", "many", "much", "even", "still", "going", "went",
    "said", "says", "say", "like", "new", "make", "made", "one", "two",
    "know", "think", "see", "look", "want", "give", "use", "find",
    "tell", "ask", "work", "seem", "feel", "try", "leave", "call",
    "need", "become", "keep", "let", "begin", "show", "hear", "play",
    "run", "move", "live", "believe", "bring", "happen", "write",
    "people", "way", "thing", "man", "day", "time", "year",
    "https", "http", "www", "com", "org", "net", "co",
    "rt", "amp", "via", "lol", "omg", "smh", "tbh", "imo",
}

# Emotional / sensationalist words that poison search queries
# These express opinion/emotion but carry zero factual search value
_EMOTIONAL_WORDS = {
    # Outrage / shock
    "shocking", "shameful", "disgusting", "outrageous", "horrifying",
    "horrific", "terrible", "horrible", "appalling", "atrocious",
    "unbelievable", "unacceptable", "despicable", "pathetic", "absurd",
    # Hype / clickbait
    "breaking", "bombshell", "explosive", "sensational", "incredible",
    "amazing", "awesome", "epic", "savage", "brutal", "insane",
    "mindblowing", "unreal", "legendary", "massive", "huge",
    # Social media filler
    "please", "share", "retweet", "viral", "trending", "thread",
    "urgent", "important", "attention", "everyone", "friends",
    "watch", "video", "check", "real", "truth", "expose",
    # Generic evaluations
    "good", "bad", "best", "worst", "great", "poor", "wonderful",
    "beautiful", "ugly", "stupid", "smart", "brilliant", "dumb",
    # Propaganda filler
    "destroying", "ruining", "failing", "betraying", "looting",
    "corrupt", "corruption", "scam", "fraud", "liar", "liars",
    "traitor", "anti", "national", "antinational",
}

# Common English words that should NOT be treated as proper nouns
# even when capitalized mid-sentence (social media often capitalizes these)
_NOT_PROPER_NOUNS = {
    # Common nouns often capitalized for emphasis
    "education", "government", "school", "schools", "college", "university",
    "hospital", "police", "army", "country", "state", "city", "district",
    "party", "election", "minister", "ministry", "department", "committee",
    "report", "data", "news", "media", "press", "public", "private",
    "official", "policy", "law", "court", "judge", "order", "notice",
    # Common verbs/adjectives capitalized for emphasis
    "improving", "working", "building", "running", "making", "taking",
    "giving", "putting", "coming", "looking", "creating", "closing",
    "opening", "shutting", "stopping", "starting", "growing", "rising",
    "falling", "killing", "fighting", "winning", "losing", "helping",
    "leading", "following", "supporting", "opposing",
    # Generic adjectives
    "only", "first", "last", "next", "free", "open", "closed",
    "under", "against", "during", "between", "every", "another",
    # Common social-media emphasis words
    "today", "now", "never", "always", "finally", "already",
    "together", "forward", "behind", "ahead",
}

# Simple stem normalization pairs (word → canonical form)
# Prevents "Govt" and "Government" from both taking keyword slots
_STEM_MAP = {
    "govt": "government", "gov": "government", "govts": "government",
    "govt's": "government",
    "edu": "education", "educ": "education",
    "min": "minister", "mins": "ministers",
    "pm": "prime minister", "cm": "chief minister",
    "parl": "parliament", "parliment": "parliament",
    "info": "information", "mgmt": "management",
    "schools": "school", "colleges": "college",
    "ministers": "minister", "hospitals": "hospital",
    "elections": "election", "policies": "policy",
    "reports": "report", "districts": "district",
    "states": "state", "cities": "city",
}


def _is_true_proper_noun(word: str) -> bool:
    """
    Check if a capitalized word is a genuine proper noun (name, place, org)
    vs just a common word capitalized for emphasis in social media.
    
    True proper nouns: Modi, Karnataka, BJP, Congress, Delhi, Rahul
    False proper nouns: Education, Improving, Government, School
    """
    lower = word.lower()
    canonical = _STEM_MAP.get(lower, lower)
    
    # If it's in our known-not-proper-noun list, it's not a real name
    if canonical in _NOT_PROPER_NOUNS or lower in _NOT_PROPER_NOUNS:
        return False
    
    # ALL CAPS short words are likely acronyms/org names (BJP, AAP, NDTV)
    if word.isupper() and len(word) <= 6:
        return True
    
    # Words that are ALL CAPS and long are probably shouting, not names
    if word.isupper() and len(word) > 6:
        return False
    
    # Title case words not in our exclusion list are likely proper nouns
    return True


def extract_search_keywords(text: str, max_keywords: int = 6) -> str:
    """
    Extract the most important keywords from a piece of text
    to form a concise search query for news APIs.

    Strategy:
      1. Remove URLs, mentions, hashtag symbols, special chars
      2. Tokenize and remove stop words + emotional filler words
      3. Deduplicate stems (Govt/Government → keep one)
      4. Categorize: true proper nouns > numbers > content nouns
      5. Allocate slots: max 4 proper nouns, reserve space for
         numbers and content words which are often more specific
      6. Return a short query string (max ~6 keywords)
    
    Example:
      Input:  "SHOCKING AND SHAMEFUL Under Congress Govt - 676
               Government schools shut down in Karnataka..."
      Output: "Congress Karnataka BJP 676 schools shut"
    """
    if not text or len(text.strip()) < 5:
        return text.strip()

    # Remove URLs
    cleaned = re.sub(r'https?://\S+', '', text)
    # Remove @mentions
    cleaned = re.sub(r'@\w+', '', cleaned)
    # Keep hashtag words but remove the # symbol
    cleaned = re.sub(r'#(\w+)', r'\1', cleaned)
    # Remove emojis and special unicode but keep digits
    cleaned = re.sub(r'[^\w\s\'-]', ' ', cleaned)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if not cleaned:
        return text[:50].strip()

    words = cleaned.split()

    # Categorize words into buckets by importance
    proper_nouns = []    # Highest priority: names, places, orgs
    numbers = []         # High priority: specific numbers ("676")
    content_words = []   # Medium: nouns, verbs that describe events

    seen_stems = set()   # Track stems to avoid duplicates

    for w in words:
        lower = w.lower()

        # Skip short words, stop words, and emotional filler
        if len(lower) < 3:
            continue
        if lower in _STOP_WORDS or lower in _EMOTIONAL_WORDS:
            continue

        # Normalize stem
        canonical = _STEM_MAP.get(lower, lower)
        if canonical in seen_stems:
            continue
        seen_stems.add(canonical)
        # Also mark the raw form so "Government" doesn't re-add after "Govt"
        seen_stems.add(lower)

        # Numbers (e.g., "676") — very specific, great for search
        if re.match(r'^\d+$', w) and len(w) >= 2:
            numbers.append(w)
        # True proper nouns only (names, places, orgs — not "Education")
        elif w[0].isupper() and len(w) > 2 and _is_true_proper_noun(w):
            proper_nouns.append(w)
        # Content words (including demoted "proper nouns" like Education)
        elif len(lower) > 2:
            content_words.append(canonical)  # use canonical form

    # Count frequencies
    proper_counts = Counter(proper_nouns)
    content_counts = Counter(content_words)

    # Build keyword list with RESERVED SLOTS to ensure diversity:
    #   - Proper nouns: up to 4 slots (names/places are important but not everything)
    #   - Numbers: up to 2 slots (very specific, crucial for verification)
    #   - Content words: remaining slots (schools, shut, etc.)
    keywords = []
    seen_final = set()

    def _add(word):
        key = _STEM_MAP.get(word.lower(), word.lower())
        if key not in seen_final and len(keywords) < max_keywords:
            keywords.append(word)
            seen_final.add(key)
            return True
        return False

    # 1. Top proper nouns (max 4 — leave room for specific terms)
    max_proper = min(4, max_keywords - 1)  # always leave at least 1 slot
    added_proper = 0
    for word, _ in proper_counts.most_common(max_proper + 2):
        if added_proper >= max_proper:
            break
        if _add(word):
            added_proper += 1

    # 2. ALL numbers (these are highly specific and crucial)
    for num in numbers:
        _add(num)

    # 3. Fill remaining with content words
    for word, _ in content_counts.most_common(max_keywords):
        if len(keywords) >= max_keywords:
            break
        _add(word)
    for word, _ in content_counts.most_common(max_keywords):
        _add(word)

    if not keywords:
        # Last resort: take first few non-trivial words
        fallback = [w for w in words if len(w) > 3 and w.lower() not in _EMOTIONAL_WORDS][:max_keywords]
        return " ".join(fallback) if fallback else text[:50].strip()

    return " ".join(keywords)


def build_search_tree(text: str) -> List[str]:
    """
    Build a tree of keyword combinations for progressive search broadening.
    
    Instead of just shrinking the same keyword list, this generates
    DIFFERENT combinations from the ranked keyword pool so each
    search attempt covers a new angle.
    
    Tree structure (example keywords: [Karnataka, Congress, BJP, 676, school, shut]):
    
        Level 0 (narrow):  "Karnataka 676 school"         ← most specific
        Level 1 (alt):     "Karnataka Congress school"     ← swap number for name
        Level 2 (shift):   "Congress BJP Karnataka"        ← names-only angle
        Level 3 (broad):   "Karnataka school"              ← 2 keywords only
        Level 4 (broadest):"Karnataka Congress"            ← broadest meaningful pair
    
    Each query is unique — no combination is repeated.
    
    Returns:
        List of query strings, ordered from most specific to broadest.
    """
    # Extract ALL keywords ranked by importance (up to 8)
    all_kw = extract_search_keywords(text, max_keywords=8)
    if not all_kw:
        return [text[:50].strip()]

    words = all_kw.split()
    if len(words) <= 2:
        return [all_kw]

    # Categorize extracted keywords
    names = []     # Proper nouns / acronyms (Karnataka, BJP, Congress)
    specifics = [] # Numbers and very specific terms (676)
    content = []   # Content words (school, shut)

    for w in words:
        if re.match(r'^\d+$', w):
            specifics.append(w)
        elif w[0].isupper():
            names.append(w)
        else:
            content.append(w)

    # Build the search tree — each level is a unique combination
    queries = []
    seen = set()

    def _add_query(combo: List[str]):
        q = " ".join(combo)
        key = frozenset(w.lower() for w in combo)
        if key not in seen and len(combo) >= 2:
            seen.add(key)
            queries.append(q)

    # --- Level 0: Most specific (name + number + content word) ---
    if names and specifics and content:
        _add_query([names[0]] + specifics[:1] + content[:1])
    elif names and specifics:
        _add_query([names[0]] + specifics[:1])
    elif names and content:
        _add_query([names[0]] + content[:2])

    # --- Level 1: Names + content (different angle) ---
    if len(names) >= 1 and content:
        _add_query([names[0]] + content[:2])
    if len(names) >= 2 and content:
        _add_query(names[:2] + content[:1])

    # --- Level 2: Names cluster (entity-focused) ---
    if len(names) >= 3:
        _add_query(names[:3])
    if len(names) >= 2:
        _add_query(names[:2])

    # --- Level 3: Number + content (fact-focused) ---
    if specifics and content:
        _add_query(specifics[:1] + content[:1] + (names[:1] if names else []))

    # --- Level 4: Broadest pairs ---
    if names and content:
        _add_query([names[0], content[0]])
    if len(names) >= 2:
        _add_query([names[0], names[-1]])

    # Fallback: if nothing was generated, just use the top keywords
    if not queries:
        queries.append(" ".join(words[:3]))
        if len(words) >= 2:
            queries.append(" ".join(words[:2]))

    return queries

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CrawledArticle:
    """Represents a single crawled news article"""
    title: str
    url: str
    source: str
    source_domain: str
    published_date: Optional[str] = None
    snippet: str = ""
    full_text: str = ""
    language: str = "en"
    crawl_source: str = ""   # which crawler found it (google, gdelt, newsapi, scrape)
    crawl_timestamp: str = ""
    
    def __post_init__(self):
        if not self.crawl_timestamp:
            self.crawl_timestamp = datetime.utcnow().isoformat()
        if not self.source_domain:
            try:
                self.source_domain = urlparse(self.url).netloc
            except Exception:
                self.source_domain = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# TRUSTED / KNOWN SOURCES REGISTRY
# ============================================================

TRUSTED_SOURCES = {
    # Major wire services
    "reuters.com", "apnews.com", "afp.com",
    # Major newspapers
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "bbc.com", "bbc.co.uk", "cnn.com",
    "aljazeera.com", "dw.com", "france24.com",
    # Fact-checkers
    "snopes.com", "factcheck.org", "politifact.com",
    "fullfact.org", "checkyourfact.com",
    # Science / Health
    "nature.com", "sciencemag.org", "who.int", "cdc.gov",
    "nih.gov", "thelancet.com", "bmj.com",
    # Indian sources
    "thehindu.com", "indianexpress.com", "ndtv.com",
    "hindustantimes.com", "livemint.com",
    "timesofindia.indiatimes.com", "indiatimes.com",
    "indiatoday.in", "india.com", "deccanherald.com",
    "scroll.in", "thewire.in", "firstpost.com",
    "opindia.com", "swarajyamag.com", "wionews.com",
    "news18.com", "zeenews.india.com", "aajtak.in",
    "theprint.in", "business-standard.com", "moneycontrol.com",
    "economictimes.indiatimes.com", "tribuneindia.com",
    "deccanchronicle.com", "newindianexpress.com",
    "freepressjournal.in", "dnaindia.com",
}

KNOWN_UNRELIABLE_SOURCES = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "worldnewsdailyreport.com", "yournewswire.com",
    "newsthump.com",  # satire
    "theonion.com",   # satire
    "babylonbee.com", # satire
}

SATIRE_SOURCES = {
    "theonion.com", "babylonbee.com", "newsthump.com",
    "clickhole.com", "borowitz-report",
}


# ============================================================
# GOOGLE NEWS RSS CRAWLER (No API Key Required)
# ============================================================

class GoogleNewsCrawler:
    """
    Crawls Google News RSS feed for real-time articles.
    No API key needed — uses the public RSS endpoint.
    """
    
    BASE_URL = "https://news.google.com/rss/search"
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
    
    async def search(self, query: str, max_results: int = 10, language: str = "en") -> List[CrawledArticle]:
        """Search Google News RSS using a keyword search tree.
        Tries progressively broader unique keyword combinations until results are found."""
        search_tree = build_search_tree(query)
        articles = []

        for i, search_query in enumerate(search_tree):
            logger.info(f"Google News tree[{i}]: '{search_query}'")

            try:
                encoded_query = quote_plus(search_query)
                url = f"{self.BASE_URL}?q={encoded_query}&hl={language}&gl=US&ceid=US:{language}"

                response = await self.client.get(url)
                response.raise_for_status()

                root = ElementTree.fromstring(response.text)

                items = root.findall(".//item")
                for item in items[:max_results]:
                    title = item.findtext("title", "")
                    link = item.findtext("link", "")
                    pub_date = item.findtext("pubDate", "")
                    description = item.findtext("description", "")
                    source_elem = item.find("source")
                    source_name = source_elem.text if source_elem is not None else ""

                    clean_desc = re.sub(r'<[^>]+>', '', description).strip()

                    articles.append(CrawledArticle(
                        title=title,
                        url=link,
                        source=source_name,
                        source_domain=urlparse(link).netloc if link else "",
                        published_date=pub_date,
                        snippet=clean_desc[:500],
                        crawl_source="google_news"
                    ))

                if articles:
                    logger.info(f"Google News: Found {len(articles)} articles at tree[{i}] '{search_query}'")
                    break  # Got results, stop searching

            except Exception as e:
                logger.error(f"Google News crawler error: {e}")
                break  # Don't retry on network errors

        if not articles:
            logger.warning("Google News: 0 articles across all search tree branches")

        return articles

    async def close(self):
        await self.client.aclose()


# ============================================================
# GDELT PROJECT CRAWLER (Free, Real-Time Global News)
# ============================================================

class GDELTCrawler:
    """
    Crawls GDELT Project DOC API for real-time global news.
    Completely free, no API key required.
    Covers 100+ languages and 200+ countries.
    """
    
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={
                "User-Agent": "OSINT-Monitor/1.0"
            }
        )
    
    async def search(self, query: str, max_results: int = 10,
                     timespan: str = "7d", language: str = "english") -> List[CrawledArticle]:
        """
        Search GDELT for news articles using a keyword search tree.
        Tries progressively broader unique keyword combinations.
        timespan: e.g., '24h', '7d', '30d'
        """
        search_tree = build_search_tree(query)
        articles = []

        for i, search_query in enumerate(search_tree):
            logger.info(f"GDELT tree[{i}]: '{search_query}'")

            try:
                params = {
                    "query": search_query,
                    "mode": "artlist",
                    "maxrecords": str(max_results),
                    "format": "json",
                    "timespan": timespan,
                    "sort": "datedesc",
                }

                response = await self.client.get(self.BASE_URL, params=params)

                # Handle rate limiting gracefully
                if response.status_code == 429:
                    logger.warning("GDELT rate limited (429). Skipping.")
                    return articles

                response.raise_for_status()

                # Validate response before JSON parsing
                content_type = response.headers.get("content-type", "")
                body = response.text.strip()

                if not body:
                    continue  # try next branch

                if "json" not in content_type:
                    if not (body.startswith('{"') or body.startswith('[{')):
                        logger.warning(f"GDELT non-JSON at tree[{i}] (starts: {body[:40]})")
                        continue

                try:
                    data = json.loads(body)
                except json.JSONDecodeError as je:
                    logger.warning(f"GDELT invalid JSON at tree[{i}]: {je}")
                    continue

                for article_data in data.get("articles", []):
                    domain = article_data.get("domain", "")
                    articles.append(CrawledArticle(
                        title=article_data.get("title", ""),
                        url=article_data.get("url", ""),
                        source=article_data.get("domain", ""),
                        source_domain=domain,
                        published_date=article_data.get("seendate", ""),
                        snippet=article_data.get("title", ""),
                        language=article_data.get("language", "English"),
                        crawl_source="gdelt"
                    ))

                if articles:
                    logger.info(f"GDELT: Found {len(articles)} articles at tree[{i}] '{search_query}'")
                    break  # Got results, stop searching

            except Exception as e:
                logger.error(f"GDELT crawler error: {e}")
                break  # Don't retry on network errors

        if not articles:
            logger.warning("GDELT: 0 articles across all search tree branches")

        return articles
    
    async def close(self):
        await self.client.aclose()


# ============================================================
# NEWSAPI CRAWLER (Optional, requires free API key)
# ============================================================

class NewsAPICrawler:
    """
    Crawls NewsAPI.org for articles.
    Requires a free API key (https://newsapi.org).
    Free tier: 100 requests/day.
    """
    
    BASE_URL = "https://newsapi.org/v2/everything"
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
    
    @property
    def available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
    
    async def search(self, query: str, max_results: int = 10,
                     days_back: int = 7) -> List[CrawledArticle]:
        """Search NewsAPI for articles"""
        if not self.available:
            logger.debug("NewsAPI key not configured, skipping")
            return []
        
        articles = []
        try:
            from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": max_results,
                "language": "en",
                "apiKey": self.api_key
            }
            
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("articles", []):
                source_name = item.get("source", {}).get("name", "")
                url = item.get("url", "")
                articles.append(CrawledArticle(
                    title=item.get("title", ""),
                    url=url,
                    source=source_name,
                    source_domain=urlparse(url).netloc if url else "",
                    published_date=item.get("publishedAt", ""),
                    snippet=item.get("description", "") or "",
                    full_text=item.get("content", "") or "",
                    crawl_source="newsapi"
                ))
            
            logger.info(f"NewsAPI: Found {len(articles)} articles for '{query}'")
            
        except Exception as e:
            logger.error(f"NewsAPI crawler error: {e}")
        
        return articles
    
    async def close(self):
        await self.client.aclose()


# ============================================================
# WEB SCRAPER (Fallback — extract article text from URLs)
# ============================================================

class ArticleScraper:
    """
    Lightweight article text extractor.
    Fetches a URL and extracts the main body text.
    Used to enrich crawled articles with full content.
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
    
    async def extract_text(self, url: str) -> str:
        """Extract main article text from a URL"""
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            html = response.text
            return self._extract_article_text(html)
        except Exception as e:
            logger.debug(f"Scraper error for {url}: {e}")
            return ""
    
    def _extract_article_text(self, html: str) -> str:
        """Extract readable text from HTML using heuristics"""
        # Remove script, style, nav, header, footer tags
        html = re.sub(r'<(script|style|nav|header|footer|aside|iframe)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract text from <p> tags (most article content lives here)
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
        
        # Clean HTML tags from extracted paragraphs
        clean_paragraphs = []
        for p in paragraphs:
            clean = re.sub(r'<[^>]+>', '', p).strip()
            # Filter out very short or ad-like paragraphs
            if len(clean) > 40:
                clean_paragraphs.append(clean)
        
        text = "\n\n".join(clean_paragraphs)
        
        # Fallback: if no <p> tags found, do a raw text extraction
        if len(text) < 100:
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            # Take a reasonable chunk
            text = text[:3000]
        
        return text[:5000]  # Cap at 5000 chars
    
    async def close(self):
        await self.client.aclose()


# ============================================================
# UNIFIED NEWS CRAWLER — Orchestrates All Sources
# ============================================================

class NewsCrawler:
    """
    Unified crawler that searches multiple sources in parallel,
    deduplicates results, enriches with full text, and returns
    a clean list of articles for credibility analysis.
    """
    
    def __init__(self):
        self.google = GoogleNewsCrawler()
        self.gdelt = GDELTCrawler()
        self.newsapi = NewsAPICrawler()
        self.scraper = ArticleScraper()
    
    async def search_news(
        self,
        query: str,
        max_results: int = 15,
        enrich_text: bool = True,
        max_enrich: int = 5,
        timespan: str = "7d",
        original_text: str = ""
    ) -> List[CrawledArticle]:
        """
        Search all available sources in parallel, deduplicate,
        semantically filter against the full original text,
        and optionally enrich top articles with full text.
        
        Args:
            query: Search query (e.g., a claim to verify)
            max_results: Max articles to return
            enrich_text: Whether to scrape full article text
            max_enrich: How many articles to enrich with full text
            timespan: GDELT time window (e.g., '24h', '7d')
            original_text: The FULL original post/claim text for
                          semantic similarity scoring. If empty,
                          falls back to keyword-based scoring.
        
        Returns:
            List of CrawledArticle objects, sorted by semantic relevance
        """
        # The text we'll use for semantic scoring (prefer full text)
        scoring_text = original_text if original_text else query

        # Run all crawlers in parallel
        google_task = self.google.search(query, max_results=max_results)
        gdelt_task = self.gdelt.search(query, max_results=max_results, timespan=timespan)
        newsapi_task = self.newsapi.search(query, max_results=max_results)
        
        results = await asyncio.gather(
            google_task, gdelt_task, newsapi_task,
            return_exceptions=True
        )
        
        # Collect all articles
        all_articles: List[CrawledArticle] = []
        source_names = ["Google News", "GDELT", "NewsAPI"]
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"{source_names[i]} failed: {result}")
            elif isinstance(result, list):
                all_articles.extend(result)
        
        logger.info(f"Total raw articles: {len(all_articles)}")
        
        # Deduplicate by title similarity
        unique_articles = self._deduplicate(all_articles)
        logger.info(f"After dedup: {len(unique_articles)} unique articles")
        
        # Score by semantic relevance to the FULL original text
        scored = self._score_articles(unique_articles, scoring_text)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter: drop articles below semantic relevance threshold
        MIN_RELEVANCE = 0.15  # cosine similarity threshold
        relevant = [(a, s) for a, s in scored if s >= MIN_RELEVANCE]

        if relevant:
            logger.info(
                f"Semantic filter: {len(relevant)}/{len(scored)} articles "
                f"above threshold {MIN_RELEVANCE} "
                f"(top={relevant[0][1]:.3f}, bottom={relevant[-1][1]:.3f})"
            )
        else:
            # If nothing passes the threshold, keep top 3 as best-effort
            relevant = scored[:3]
            logger.warning(
                f"No articles above semantic threshold {MIN_RELEVANCE}, "
                f"keeping top 3 (scores: {[round(s, 3) for _, s in relevant]})"
            )

        top_articles = [a for a, _ in relevant[:max_results]]
        
        # Enrich top articles with full text
        if enrich_text and top_articles:
            articles_to_enrich = top_articles[:max_enrich]
            enrich_tasks = [self.scraper.extract_text(a.url) for a in articles_to_enrich]
            enriched_texts = await asyncio.gather(*enrich_tasks, return_exceptions=True)
            
            for article, text_result in zip(articles_to_enrich, enriched_texts):
                if isinstance(text_result, str) and text_result:
                    article.full_text = text_result
        
        return top_articles
    
    def _deduplicate(self, articles: List[CrawledArticle]) -> List[CrawledArticle]:
        """Remove duplicate articles based on title similarity"""
        seen_hashes = set()
        unique = []
        
        for article in articles:
            # Normalize title for comparison
            normalized = re.sub(r'[^\w\s]', '', article.title.lower()).strip()
            title_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
            
            if title_hash not in seen_hashes:
                seen_hashes.add(title_hash)
                unique.append(article)
        
        return unique
    
    def _score_articles(self, articles: List[CrawledArticle], full_text: str) -> List[tuple]:
        """
        Score articles by semantic relevance to the FULL original text.
        Uses SBERT cosine similarity (the same model used for topics).
        Falls back to keyword overlap if SBERT is unavailable.
        """
        embedder = _get_embedder()

        if embedder is not None:
            return self._score_semantic(articles, full_text, embedder)
        else:
            logger.warning("SBERT embedder not available, falling back to keyword scoring")
            return self._score_keyword_fallback(articles, full_text)

    def _score_semantic(self, articles: List[CrawledArticle],
                        full_text: str, embedder) -> List[tuple]:
        """
        Score articles using SBERT cosine similarity against the full
        original post text. This catches semantic matches even when
        exact keywords differ (e.g. '676 schools shut' vs
        'Karnataka government closes 676 educational institutions').
        """
        # Encode the full original text
        text_embedding = embedder.encode(full_text[:512], convert_to_numpy=True)

        # Encode all article titles + snippets in a single batch
        article_texts = [
            (a.title + ". " + a.snippet)[:256] for a in articles
        ]
        if not article_texts:
            return []

        article_embeddings = embedder.encode(article_texts, convert_to_numpy=True,
                                             batch_size=32)

        scored = []
        for i, article in enumerate(articles):
            # Cosine similarity
            cos_sim = float(np.dot(text_embedding, article_embeddings[i]) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(article_embeddings[i]) + 1e-8
            ))

            # Small boost for trusted sources
            domain = article.source_domain.lower()
            if any(t in domain for t in TRUSTED_SOURCES):
                cos_sim += 0.05
            # Small penalty for unreliable sources
            if any(u in domain for u in KNOWN_UNRELIABLE_SOURCES):
                cos_sim -= 0.10

            scored.append((article, cos_sim))

        # Log top scores for debugging
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        top_3 = [(a.title[:60], round(s, 3)) for a, s in scored_sorted[:3]]
        logger.info(f"Semantic scores (top 3): {top_3}")

        return scored

    def _score_keyword_fallback(self, articles: List[CrawledArticle],
                                query: str) -> List[tuple]:
        """Keyword-based scoring fallback when SBERT is unavailable."""
        query_words = set(query.lower().split())
        scored = []
        
        for article in articles:
            score = 0.0
            title_lower = article.title.lower()
            snippet_lower = article.snippet.lower()
            
            # Title word overlap
            title_words = set(title_lower.split())
            overlap = len(query_words & title_words) / max(len(query_words), 1)
            score += overlap * 3.0
            
            # Snippet word overlap
            snippet_words = set(snippet_lower.split())
            snippet_overlap = len(query_words & snippet_words) / max(len(query_words), 1)
            score += snippet_overlap * 1.0
            
            # Boost trusted sources
            domain = article.source_domain.lower()
            for trusted in TRUSTED_SOURCES:
                if trusted in domain:
                    score += 2.0
                    break
            
            # Penalize known unreliable
            for unreliable in KNOWN_UNRELIABLE_SOURCES:
                if unreliable in domain:
                    score -= 3.0
                    break
            
            # Recency boost (if date available)
            if article.published_date:
                try:
                    # Try common date formats
                    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%a, %d %b %Y %H:%M:%S %Z",
                                "%Y%m%dT%H%M%SZ", "%Y-%m-%d"]:
                        try:
                            pub_dt = datetime.strptime(article.published_date, fmt)
                            days_old = (datetime.utcnow() - pub_dt).days
                            if days_old < 1:
                                score += 1.5
                            elif days_old < 3:
                                score += 1.0
                            elif days_old < 7:
                                score += 0.5
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            scored.append((article, score))
        
        return scored
    
    async def close(self):
        """Cleanup all HTTP clients"""
        await asyncio.gather(
            self.google.close(),
            self.gdelt.close(),
            self.newsapi.close(),
            self.scraper.close(),
            return_exceptions=True
        )
    
    def get_source_info(self, domain: str) -> Dict[str, Any]:
        """Get trust information about a source domain"""
        domain_lower = domain.lower()
        
        is_trusted = any(t in domain_lower for t in TRUSTED_SOURCES)
        is_unreliable = any(u in domain_lower for u in KNOWN_UNRELIABLE_SOURCES)
        is_satire = any(s in domain_lower for s in SATIRE_SOURCES)
        
        if is_satire:
            return {"trust_level": "satire", "score": 0.1, "note": "Known satire publication"}
        elif is_unreliable:
            return {"trust_level": "unreliable", "score": 0.15, "note": "Known unreliable source"}
        elif is_trusted:
            return {"trust_level": "trusted", "score": 0.9, "note": "Established news organization"}
        else:
            return {"trust_level": "unknown", "score": 0.5, "note": "Source not in registry"}
