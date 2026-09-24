"""
Chunk each source's text, embed with fastembed (ONNX-based, no PyTorch),
and store in a per-session FAISS index with source_id metadata.
fastembed uses onnxruntime under the hood — no torch.dll required.
"""
import os
import pickle
from typing import List

import numpy as np
import faiss
from fastembed import TextEmbedding

from config import get_settings

settings = get_settings()
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        # BAAI/bge-small-en-v1.5 is compact (33MB), fast, ONNX-native
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index(scraped_sources: List[dict], session_id: str) -> dict:
    """
    Build a FAISS index for the session.
    Returns metadata dict saved alongside the index.

    metadata structure:
      {
        "chunks": [{"source_id", "url", "title", "text"}, ...],
        "sources": [{"source_id", "url", "title"}, ...]
      }
    """
    model = _get_model()
    all_chunks = []  # list of {source_id, url, title, text}
    all_texts = []

    for src in scraped_sources:
        chunks = _chunk_text(src["text"], settings.chunk_size, settings.chunk_overlap)
        for c in chunks:
            all_chunks.append(
                {
                    "source_id": src["source_id"],
                    "url": src["url"],
                    "title": src["title"],
                    "text": c,
                }
            )
            all_texts.append(c)

    print(f"[embedder] Embedding {len(all_texts)} chunks with fastembed...")
    # fastembed returns a generator of numpy arrays
    embeddings_gen = model.embed(all_texts, batch_size=32)
    embeddings = np.array(list(embeddings_gen), dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product after L2-norm = cosine similarity
    index.add(embeddings)

    # Persist
    os.makedirs(settings.faiss_index_dir, exist_ok=True)
    index_path = os.path.join(settings.faiss_index_dir, f"{session_id}.index")
    meta_path = os.path.join(settings.faiss_index_dir, f"{session_id}.meta.pkl")

    faiss.write_index(index, index_path)
    meta = {
        "chunks": all_chunks,
        "sources": [
            {"source_id": s["source_id"], "url": s["url"], "title": s["title"]}
            for s in scraped_sources
        ],
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"[embedder] Index saved: {index_path}")
    return meta


def load_index(session_id: str):
    """Load FAISS index and metadata for a session. Returns (index, meta)."""
    index_path = os.path.join(settings.faiss_index_dir, f"{session_id}.index")
    meta_path = os.path.join(settings.faiss_index_dir, f"{session_id}.meta.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index found for session {session_id}")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta
