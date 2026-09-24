"""
Scrape and clean text content from URLs.
Uses trafilatura as primary extractor, falls back to BeautifulSoup.
"""
import re
import time
import random
from typing import Optional

import httpx
import trafilatura
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def scrape_url(url: str, timeout: int = 20) -> Optional[str]:
    """Fetch and extract main content from `url`. Returns None on failure."""
    try:
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        print(f"[scraper] Failed to fetch {url}: {exc}")
        return None

    # trafilatura — best-effort main content extraction
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if text and len(text) > 200:
        return _clean_text(text)

    # BS4 fallback — strip scripts/styles, get body text
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    body = soup.get_text(separator="\n")
    if body and len(body) > 200:
        return _clean_text(body)

    return None


def scrape_sources(sources: list[dict]) -> list[dict]:
    """
    Given a list of {url, title, ...} dicts, scrape each.
    Returns list of {source_id, url, title, text, char_count}.
    Labels are Source A, B, C, D.
    """
    labels = ["A", "B", "C", "D"]
    result = []
    for i, src in enumerate(sources):
        label = labels[i] if i < len(labels) else f"Src{i}"
        time.sleep(random.uniform(0.5, 1.5))  # polite crawl delay
        text = scrape_url(src["url"])
        if not text:
            print(f"[scraper] Could not extract content from {src['url']}")
            continue
        result.append(
            {
                "source_id": f"Source {label}",
                "url": src["url"],
                "title": src.get("title", src["url"]),
                "text": text,
                "char_count": len(text),
            }
        )
    return result
