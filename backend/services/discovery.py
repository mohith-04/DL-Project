"""
Auto-discover 2-3 related URLs for a given topic.

Strategy (in order of priority):
1. Wikipedia search API  — free, reliable, no rate limits
2. Google search via requests with fallback User-Agent rotation
3. DuckDuckGo (v3 API as last resort)

All discovered URLs come from different domains than the primary URL.
"""
import re
import time
import random
from urllib.parse import urlparse, urlencode, quote_plus
from typing import List

import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


def _topic_from_url(url: str) -> str:
    """Heuristically extract topic keywords from URL path."""
    path = urlparse(url).path
    path = re.sub(r"\.(html?|php|aspx?)$", "", path, flags=re.I)
    parts = re.split(r"[/_\-]+", path)
    meaningful = [p for p in parts if len(p) > 3]
    return " ".join(meaningful[-5:]) if meaningful else ""


SKIP_DOMAINS = {
    "youtube.com", "reddit.com", "twitter.com", "x.com",
    "quora.com", "facebook.com", "instagram.com", "linkedin.com",
    "pinterest.com", "tiktok.com",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Strategy 1: Wikipedia Search API (most reliable)
# ---------------------------------------------------------------------------

def _search_wikipedia(topic: str, n: int) -> List[dict]:
    """
    Use Wikipedia's opensearch + summary API to find related articles.
    Returns list of {url, title, snippet}.
    """
    results = []
    try:
        params = {
            "action": "opensearch",
            "search": topic,
            "limit": n + 3,
            "namespace": 0,
            "format": "json",
        }
        resp = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        # opensearch returns [query, [titles], [descriptions], [urls]]
        titles = data[1]
        descriptions = data[2]
        urls = data[3]
        for title, desc, url in zip(titles, descriptions, urls):
            results.append({"url": url, "title": title, "snippet": desc})
    except Exception as e:
        print(f"[discovery] Wikipedia search failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy 2: Google search (scrape)
# ---------------------------------------------------------------------------

def _search_google(topic: str, n: int) -> List[dict]:
    """
    Scrape Google search results page for URLs.
    Fragile but works as a fallback.
    """
    results = []
    try:
        query = quote_plus(f"{topic} tutorial explanation site:geeksforgeeks.org OR site:towardsdatascience.com OR site:javatpoint.com OR site:ibm.com OR site:aws.amazon.com")
        url = f"https://www.google.com/search?q={query}&num=10"
        resp = httpx.get(url, headers=HEADERS, timeout=15, follow_redirects=True)
        # Extract URLs from the HTML
        found = re.findall(r'href="(https?://[^"]+)"', resp.text)
        for href in found:
            if any(skip in href for skip in ["google.com", "youtube.com", "accounts."]):
                continue
            results.append({"url": href, "title": href, "snippet": ""})
            if len(results) >= n * 3:
                break
    except Exception as e:
        print(f"[discovery] Google search failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy 3: DuckDuckGo HTML endpoint (no API key, no impersonation)
# ---------------------------------------------------------------------------

def _search_ddg_html(topic: str, n: int) -> List[dict]:
    """
    Use DuckDuckGo's lite HTML endpoint — doesn't require impersonation.
    """
    results = []
    try:
        params = {"q": f"{topic} tutorial explanation", "kl": "us-en"}
        resp = httpx.get(
            "https://lite.duckduckgo.com/lite/",
            params=params,
            headers=HEADERS,
            timeout=15,
            follow_redirects=True,
        )
        # Extract result links from the HTML
        found = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>', resp.text)
        for href in found:
            if "duckduckgo.com" in href:
                continue
            results.append({"url": href, "title": href, "snippet": ""})
            if len(results) >= n * 3:
                break
    except Exception as e:
        print(f"[discovery] DDG-HTML search failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy 4: Curated trusted sources per topic keyword
# ---------------------------------------------------------------------------

CURATED_SOURCES = {
    "retrieval": [
        {"url": "https://www.pinecone.io/learn/retrieval-augmented-generation/", "title": "RAG - Pinecone", "snippet": ""},
        {"url": "https://aws.amazon.com/what-is/retrieval-augmented-generation/", "title": "What is RAG? - AWS", "snippet": ""},
        {"url": "https://www.ibm.com/topics/retrieval-augmented-generation", "title": "RAG - IBM", "snippet": ""},
    ],
    "dijkstra": [
        {"url": "https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/", "title": "Dijkstra's - GFG", "snippet": ""},
        {"url": "https://www.programiz.com/dsa/dijkstra-algorithm", "title": "Dijkstra's - Programiz", "snippet": ""},
    ],
    "machine learning": [
        {"url": "https://www.ibm.com/topics/machine-learning", "title": "Machine Learning - IBM", "snippet": ""},
        {"url": "https://www.geeksforgeeks.org/machine-learning/", "title": "Machine Learning - GFG", "snippet": ""},
    ],
    "neural network": [
        {"url": "https://www.ibm.com/topics/neural-networks", "title": "Neural Networks - IBM", "snippet": ""},
        {"url": "https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/", "title": "Neural Networks - GFG", "snippet": ""},
    ],
    "transformer": [
        {"url": "https://www.ibm.com/topics/transformer-model", "title": "Transformer Model - IBM", "snippet": ""},
        {"url": "https://www.geeksforgeeks.org/transformer-neural-network/", "title": "Transformer - GFG", "snippet": ""},
    ],
}


def _curated_fallback(topic: str, primary_domain: str, n: int) -> List[dict]:
    """Return curated sources if any keyword matches."""
    topic_lower = topic.lower()
    for kw, sources in CURATED_SOURCES.items():
        if kw in topic_lower:
            return [s for s in sources if _extract_domain(s["url"]) != primary_domain][:n]
    return []


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def discover_related_urls(
    primary_url: str,
    topic_hint: str = "",
    n: int = 3,
) -> List[dict]:
    """
    Returns up to `n` dicts: {url, title, snippet}
    from domains different from the primary URL.
    Tries Wikipedia → Google → DDG-HTML → curated fallback.
    """
    primary_domain = _extract_domain(primary_url)
    topic = topic_hint if topic_hint else _topic_from_url(primary_url)
    if not topic:
        topic = primary_url

    results = []
    seen_domains = {primary_domain, "en.wikipedia.org"}  # exclude Wikipedia (likely primary)

    # If primary IS Wikipedia, allow other Wikipedia pages but prefer external
    if "wikipedia.org" in primary_url:
        seen_domains = {primary_domain}

    def _add_results(candidates: List[dict]):
        for r in candidates:
            url = r.get("url", "")
            if not url or not url.startswith("http"):
                continue
            domain = _extract_domain(url)
            if domain in seen_domains:
                continue
            if any(skip in domain for skip in SKIP_DOMAINS):
                continue
            seen_domains.add(domain)
            results.append(r)
            if len(results) >= n:
                return

    # 1. Wikipedia search (best for academic/CS topics)
    print(f"[discovery] Trying Wikipedia search for: {topic!r}")
    wiki_results = _search_wikipedia(topic, n + 3)
    # Wikipedia results are also Wikipedia — useful only if primary is NOT Wikipedia
    if "wikipedia.org" not in primary_url:
        _add_results(wiki_results)

    # 2. Google scrape (if still need more)
    if len(results) < n:
        print(f"[discovery] Trying Google scrape for: {topic!r}")
        _add_results(_search_google(topic, n))
        time.sleep(1)

    # 3. DDG HTML lite (if still need more)
    if len(results) < n:
        print(f"[discovery] Trying DDG-HTML for: {topic!r}")
        _add_results(_search_ddg_html(topic, n))

    # 4. Curated fallback (guaranteed results for common topics)
    if len(results) < n:
        print(f"[discovery] Using curated fallback for: {topic!r}")
        _add_results(_curated_fallback(topic, primary_domain, n))

    print(f"[discovery] Found {len(results)} related URL(s) for {topic!r}")
    return results[:n]
