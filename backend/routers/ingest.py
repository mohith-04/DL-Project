"""
POST /api/ingest
Accepts a primary URL + topic name, discovers related sources,
scrapes all, builds FAISS index, returns session_id + source list.
"""
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional

from services.discovery import discover_related_urls
from services.scraper import scrape_sources
from services.embedder import build_index
from config import get_settings

router = APIRouter()
settings = get_settings()


class IngestRequest(BaseModel):
    topic_url: str
    topic_name: Optional[str] = ""
    n_related: Optional[int] = 3


class SourceInfo(BaseModel):
    source_id: str
    url: str
    title: str
    char_count: int


class IngestResponse(BaseModel):
    session_id: str
    topic_name: str
    sources: list[SourceInfo]
    total_chunks: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """
    1. Auto-discover related URLs
    2. Scrape all sources
    3. Embed + index
    4. Return session_id
    """
    session_id = str(uuid.uuid4())
    topic_hint = req.topic_name or ""

    # Step 1: Discover related URLs
    print(f"[ingest] Discovering related URLs for: {req.topic_url}")
    try:
        related = discover_related_urls(
            primary_url=req.topic_url,
            topic_hint=topic_hint,
            n=req.n_related or 3,
        )
    except Exception as e:
        print(f"[ingest] Discovery failed: {e}")
        related = []

    # Build full source list: primary first, then discovered
    all_sources = [{"url": req.topic_url, "title": topic_hint or req.topic_url}] + related
    all_sources = all_sources[: settings.max_sources]

    # Step 2: Scrape
    print(f"[ingest] Scraping {len(all_sources)} sources...")
    scraped = scrape_sources(all_sources)

    if not scraped:
        raise HTTPException(status_code=422, detail="Could not extract content from any source.")

    # Step 3: Build index
    meta = build_index(scraped, session_id)

    sources_out = [
        SourceInfo(
            source_id=s["source_id"],
            url=s["url"],
            title=s["title"],
            char_count=s["char_count"],
        )
        for s in scraped
    ]

    total_chunks = len(meta["chunks"])
    topic_name = topic_hint or req.topic_url

    return IngestResponse(
        session_id=session_id,
        topic_name=topic_name,
        sources=sources_out,
        total_chunks=total_chunks,
    )
