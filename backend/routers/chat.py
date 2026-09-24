"""
POST /api/chat
Accepts session_id + question.
Runs per-source retrieval → classification → synthesis → returns structured response.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from services.retriever import retrieve_per_source
from services.classifier import classify_sources
from services.synthesizer import synthesize_answer
from services.embedder import load_index

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    question: str
    top_k: Optional[int] = None


class ChunkInfo(BaseModel):
    source_id: str
    url: str
    title: str
    text: str
    score: float


class SourceSummary(BaseModel):
    source_id: str
    url: str
    title: str
    summary: str  # from classifier per_source_summary


class ClassificationResult(BaseModel):
    classification: str  # agree | partial | conflict
    confidence: float
    reasoning: str
    key_differences: str


class ChatResponse(BaseModel):
    answer: str
    classification: ClassificationResult
    source_summaries: List[SourceSummary]
    retrieved_chunks: Dict[str, List[ChunkInfo]]
    session_id: str
    question: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Validate session exists
    try:
        _, meta = load_index(req.session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found. Please ingest first.")

    # Step 1: Per-source retrieval
    per_source = retrieve_per_source(
        session_id=req.session_id,
        question=req.question,
        top_k=req.top_k,
    )

    # Step 2: Classification (Groq call #1)
    classification_raw = classify_sources(
        question=req.question,
        per_source_chunks=per_source,
    )

    # Step 3: Synthesis (Groq call #2)
    answer = synthesize_answer(
        question=req.question,
        per_source_chunks=per_source,
        classification=classification_raw,
    )

    # Build source summaries from classification output
    source_meta = {s["source_id"]: s for s in meta["sources"]}
    per_src_summary = classification_raw.get("per_source_summary", {})
    source_summaries = []
    for sid, summary in per_src_summary.items():
        sm = source_meta.get(sid, {"url": "", "title": sid})
        source_summaries.append(
            SourceSummary(
                source_id=sid,
                url=sm.get("url", ""),
                title=sm.get("title", sid),
                summary=summary,
            )
        )

    # Flatten chunks for response
    retrieved_chunks_out = {}
    for sid, chunks in per_source.items():
        retrieved_chunks_out[sid] = [
            ChunkInfo(
                source_id=c["source_id"],
                url=c["url"],
                title=c["title"],
                text=c["text"],
                score=c["score"],
            )
            for c in chunks
        ]

    return ChatResponse(
        answer=answer,
        classification=ClassificationResult(
            classification=classification_raw.get("classification", "agree"),
            confidence=float(classification_raw.get("confidence", 0.8)),
            reasoning=classification_raw.get("reasoning", ""),
            key_differences=classification_raw.get("key_differences", "None"),
        ),
        source_summaries=source_summaries,
        retrieved_chunks=retrieved_chunks_out,
        session_id=req.session_id,
        question=req.question,
    )
