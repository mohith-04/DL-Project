"""
Auto-Benchmark Runner for Multi-Source AI Tutor.

Automatically creates 5 domain sessions (one for each academic category)
by ingesting authoritative multi-source topics, generates eval/sessions_map.json,
and executes the 100-question evaluation suite.

Usage:
    # Ingest all 5 categories and run full 100-question eval:
    python eval/auto_benchmark.py

    # Only ingest the 5 categories (save sessions_map.json without running eval yet):
    python eval/auto_benchmark.py --ingest_only

    # Run eval using existing sessions_map.json:
    python eval/auto_benchmark.py --eval_only
"""
import argparse
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from eval.run_eval import run_eval

CATEGORY_TOPICS = {
    "RAG": {
        "topic_name": "Retrieval-Augmented Generation",
        "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    },
    "ML": {
        "topic_name": "Machine Learning Foundations",
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
    },
    "DL": {
        "topic_name": "Deep Learning & Neural Architectures",
        "url": "https://en.wikipedia.org/wiki/Deep_learning",
    },
    "Transformers": {
        "topic_name": "Transformer Architecture & Large Language Models",
        "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    },
    "MultiSource_Conflict": {
        "topic_name": "Multi-Document Cross-Source Synthesis",
        "url": "https://en.wikipedia.org/wiki/Multi-document_summarization",
    },
}


def ingest_category_sessions(base_url: str = "http://localhost:8000") -> dict:
    """Ingest a multi-source topic for each of the 5 categories and return session map."""
    print("\n" + "=" * 70)
    print("INGESTING 5 DOMAIN KNOWLEDGE SESSIONS FOR 100-QUESTION BENCHMARK")
    print("=" * 70)

    session_map = {}
    client = httpx.Client(timeout=180.0)

    for cat, info in CATEGORY_TOPICS.items():
        print(f"\n[ingest] Ingesting Category '{cat}'...")
        print(f"         Topic: {info['topic_name']}")
        print(f"         URL:   {info['url']}")

        try:
            resp = client.post(
                f"{base_url}/api/ingest",
                json={
                    "topic_url": info["url"],
                    "topic_name": info["topic_name"],
                    "n_related": 3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            sid = data["session_id"]
            num_chunks = data.get("total_chunks", 0)
            num_sources = len(data.get("sources", []))
            session_map[cat] = sid
            print(f"  ✅ SUCCESS: Session ID = {sid} ({num_sources} sources, {num_chunks} chunks)")
        except Exception as e:
            print(f"  ❌ FAILED to ingest '{cat}': {e}")
            raise

        time.sleep(1.0)

    return session_map


def main():
    parser = argparse.ArgumentParser(description="Automated 5-Category Benchmark for Multi-Source AI Tutor")
    parser.add_argument("--base_url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--sessions_file", default="eval/sessions_map.json", help="Path to sessions_map.json")
    parser.add_argument("--ingest_only", action="store_true", help="Only perform ingestion and save session map")
    parser.add_argument("--eval_only", action="store_true", help="Skip ingestion and run evaluation using sessions_map.json")
    parser.add_argument("--skip_bertscore", action="store_true", default=True, help="Skip heavy BERTScore")
    parser.add_argument("--n_questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted evaluation")
    args = parser.parse_args()

    sessions_path = Path(args.sessions_file)
    if not sessions_path.exists():
        for candidate in [
            Path(__file__).resolve().parent / sessions_path.name,
            Path(__file__).resolve().parent / args.sessions_file,
            Path(__file__).resolve().parent.parent / args.sessions_file,
        ]:
            if candidate.exists():
                sessions_path = candidate
                break

    if args.eval_only:
        if not sessions_path.exists():
            raise FileNotFoundError(f"Sessions file '{sessions_path}' not found. Run without --eval_only first.")
        with open(sessions_path, "r", encoding="utf-8") as f:
            session_map = json.load(f)
        print(f"[auto_benchmark] Loaded existing session map from {sessions_path}")
    else:
        session_map = ingest_category_sessions(base_url=args.base_url)
        sessions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sessions_path, "w", encoding="utf-8") as f:
            json.dump(session_map, f, indent=2)
        print(f"\n[auto_benchmark] Session map saved to {sessions_path.resolve()}")

    if args.ingest_only:
        print("[auto_benchmark] --ingest_only specified. Exiting.")
        return

    # Run full evaluation using the multi-session map
    run_eval(
        session_map=session_map,
        eval_set_path="eval/eval_set.json",
        base_url=args.base_url,
        skip_bertscore=args.skip_bertscore,
        n_questions=args.n_questions,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
