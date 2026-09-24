"""
Per-source top-k retrieval from FAISS.
Each source gets its own retrieval budget (top_k_per_source),
so no single dominant source can crowd out others.
Uses fastembed (ONNX) — no PyTorch dependency.
"""
import numpy as np
import faiss
from typing import List, Dict

from config import get_settings
from services.embedder import load_index, _get_model

settings = get_settings()


def retrieve_per_source(
    session_id: str,
    question: str,
    top_k: int | None = None,
) -> Dict[str, List[dict]]:
    """
    Returns dict: source_id -> list of top-k chunk dicts {source_id, url, title, text, score}.
    Each source independently contributes top_k chunks.
    """
    top_k = top_k or settings.top_k_per_source
    index, meta = load_index(session_id)
    model = _get_model()

    # Encode & normalize query using fastembed
    q_vec = list(model.embed([question]))
    q_vec = np.array(q_vec, dtype=np.float32)
    faiss.normalize_L2(q_vec)

    # Global search — retrieve enough candidates
    n_chunks = len(meta["chunks"])
    k_global = min(n_chunks, top_k * len(meta["sources"]) * 3)
    scores, indices = index.search(q_vec, k_global)

    scores = scores[0]
    indices = indices[0]

    # Group by source, keep top-k per source
    per_source: Dict[str, List[dict]] = {s["source_id"]: [] for s in meta["sources"]}
    per_source_count: Dict[str, int] = {s["source_id"]: 0 for s in meta["sources"]}

    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        chunk = meta["chunks"][idx]
        sid = chunk["source_id"]
        if per_source_count.get(sid, 0) < top_k:
            per_source[sid].append(
                {
                    **chunk,
                    "score": float(score),
                }
            )
            per_source_count[sid] = per_source_count.get(sid, 0) + 1

    return per_source
