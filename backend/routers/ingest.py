"""
POST /api/ingest         — URL-based ingestion
POST /api/ingest_files   — File upload ingestion (PDF / TXT / MD)
POST /api/ingest_hybrid  — Combined ingestion (Web URL + Local Files simultaneously)
"""
import uuid
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from services.discovery import discover_related_urls
from services.scraper import scrape_sources
from services.embedder import build_index
from services.pdf_parser import extract_text_from_file
from config import get_settings

router = APIRouter()
settings = get_settings()

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


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
    URL-only ingestion:
    1. Auto-discover related URLs
    2. Scrape all sources
    3. Embed + index in FAISS
    4. Return session_id
    """
    session_id = str(uuid.uuid4())
    topic_hint = req.topic_name or ""

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

    all_sources = [{"url": req.topic_url, "title": topic_hint or req.topic_url}] + related
    all_sources = all_sources[: settings.max_sources]

    print(f"[ingest] Scraping {len(all_sources)} sources...")
    scraped = scrape_sources(all_sources)

    if not scraped:
        raise HTTPException(status_code=422, detail="Could not extract content from any source.")

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

    return IngestResponse(
        session_id=session_id,
        topic_name=topic_hint or req.topic_url,
        sources=sources_out,
        total_chunks=len(meta["chunks"]),
    )


@router.post("/ingest_files", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(..., description="PDF, TXT, or MD files to ingest"),
    topic_name: Optional[str] = Form(""),
):
    """
    Files-only ingestion:
    Upload 1-4 PDF/TXT/MD files. Extracts text and builds FAISS index.
    """
    return await ingest_hybrid(files=files, topic_url="", topic_name=topic_name)


@router.post("/ingest_hybrid", response_model=IngestResponse)
async def ingest_hybrid(
    files: List[UploadFile] = File(default=[], description="Uploaded PDF/TXT/MD files"),
    topic_url: Optional[str] = Form("", description="Primary web source URL"),
    topic_name: Optional[str] = Form("", description="Optional topic description"),
    n_related: Optional[int] = Form(2, description="Number of related web sources to discover"),
):
    """
    Hybrid Multi-Source Ingestion:
    Accepts BOTH a web URL (with auto-discovery) AND local document files (PDF/TXT/MD),
    extracts content from all of them, and indexes them into a unified FAISS session.
    """
    valid_files = [f for f in (files or []) if f.filename and f.filename.strip()]
    has_url = bool(topic_url and topic_url.strip())
    has_files = bool(valid_files)

    if not has_url and not has_files:
        raise HTTPException(
            status_code=422,
            detail="Please provide at least one source (enter a Web URL, upload local files, or both).",
        )

    session_id = str(uuid.uuid4())
    combined_sources = []
    label_idx = 0

    # 1. Process Web URL & Discover Related Sources (if URL provided)
    if has_url:
        clean_url = topic_url.strip()
        topic_hint = topic_name.strip() if topic_name else ""
        print(f"[ingest_hybrid] Ingesting Web Source: {clean_url}")

        # If files are also attached, limit discovered web sources so we don't blow up max_sources
        n_rel = min(n_related or 1, 1) if has_files else (n_related or 2)
        try:
            related = discover_related_urls(
                primary_url=clean_url,
                topic_hint=topic_hint,
                n=n_rel,
            )
        except Exception as e:
            print(f"[ingest_hybrid] Web discovery failed: {e}")
            related = []

        all_web = [{"url": clean_url, "title": topic_hint or clean_url}] + related
        scraped_web = scrape_sources(all_web)

        for src in scraped_web:
            if label_idx < len(LABELS):
                src["source_id"] = f"Source {LABELS[label_idx]}"
                label_idx += 1
                combined_sources.append(src)

    # 2. Process Local Files (PDF / TXT / MD) (if files provided)
    if has_files:
        print(f"[ingest_hybrid] Ingesting {len(valid_files)} local files...")
        for upload in valid_files:
            if label_idx >= len(LABELS):
                break
            fname = upload.filename
            file_bytes = await upload.read()
            print(f"[ingest_hybrid] Extracting text from {fname} ({len(file_bytes)} bytes)")
            text = extract_text_from_file(fname, file_bytes)

            if text and len(text.strip()) > 50:
                combined_sources.append(
                    {
                        "source_id": f"Source {LABELS[label_idx]}",
                        "url": f"file://{fname}",
                        "title": f"Doc: {fname}",
                        "text": text,
                        "char_count": len(text),
                    }
                )
                label_idx += 1
            else:
                print(f"[ingest_hybrid] Could not extract text from {fname}")

    if not combined_sources:
        raise HTTPException(
            status_code=422,
            detail="Could not extract readable content from any provided URL or uploaded file.",
        )

    # 3. Build Unified FAISS Index
    print(f"[ingest_hybrid] Building FAISS index across {len(combined_sources)} total sources...")
    meta = build_index(combined_sources, session_id)

    sources_out = [
        SourceInfo(
            source_id=s["source_id"],
            url=s["url"],
            title=s["title"],
            char_count=s["char_count"],
        )
        for s in combined_sources
    ]

    # Generate friendly topic title
    if topic_name and topic_name.strip():
        final_topic = topic_name.strip()
    elif has_url:
        final_topic = topic_url.strip()
    elif combined_sources:
        final_topic = combined_sources[0]["title"]
    else:
        final_topic = "Multi-Source Session"

    return IngestResponse(
        session_id=session_id,
        topic_name=final_topic,
        sources=sources_out,
        total_chunks=len(meta["chunks"]),
    )
