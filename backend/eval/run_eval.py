"""
Evaluation pipeline for the Multi-Source AI Tutor.
Measures:
  (a) Retrieval quality per source (avg similarity score)
  (b) Classification accuracy (agree/partial/conflict vs. ground truth)
  (c) Answer quality via LLM-as-judge rubric

Usage:
    python eval/run_eval.py --session_id <sid> --eval_set eval/eval_set.json
"""
import argparse
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from groq import Groq
from config import get_settings

settings = get_settings()

JUDGE_SYSTEM = """
You are an expert RAG (Retrieval-Augmented Generation) evaluator.
Score the following AI tutor system on a 0-100 scale and return JSON:
{
  "accuracy": 0-100,           // Is the answer factually correct compared to the ground truth?
  "faithfulness": 0-100,       // Are all claims in the answer supported by the retrieved context? (No hallucinations)
  "context_recall": 0-100,     // Does the retrieved context contain all the information needed from the ground truth?
  "context_precision": 0-100,  // Is the retrieved context highly relevant to the question, without unnecessary noise?
  "provenance": 0-100,         // Does the answer properly cite sources inline (e.g. [Source A])?
  "overall": 0-100,            // Overall quality of the response
  "justification": "brief explanation of scores"
}
Scores should be an integer between 0 and 100.
IMPORTANT: Output a single, perfectly valid JSON object. Do not add trailing commas or extra closing braces (}).
"""


def llm_judge(question: str, answer: str, ground_truth: str, context: str) -> dict:
    import groq
    import re
    client = Groq(api_key=settings.groq_api_key)
    user_msg = f"Question: {question}\n\nGround Truth: {ground_truth}\n\nRetrieved Context:\n{context}\n\nAI Tutor Answer:\n{answer}"
    
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model_fallback,  # use faster model for judge
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except groq.BadRequestError as e:
        # If the model slightly messed up the JSON (e.g., trailing }}), Groq throws an error 
        # but provides the failed string. We can catch it and fix it manually!
        try:
            err_body = e.response.json()
            failed_text = err_body.get("error", {}).get("failed_generation", "")
            if failed_text:
                # Fix double closing braces at the end
                cleaned = re.sub(r'\}\}\s*$', '}', failed_text.strip())
                return json.loads(cleaned)
        except Exception:
            pass
        raise  # re-raise if we couldn't fix it


def run_eval(session_id: str, eval_set_path: str, base_url: str = "http://localhost:8000"):
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    results = []
    classification_correct = 0
    total = len(eval_set)

    for item in eval_set:
        print(f"\n[eval] Running: {item['id']} — {item['question'][:60]}...")
        try:
            resp = httpx.post(
                f"{base_url}/api/chat",
                json={"session_id": session_id, "question": item["question"]},
                timeout=120,
            )
            resp.raise_for_status()
            chat_resp = resp.json()
        except Exception as e:
            print(f"[eval] FAILED: {e}")
            results.append({"id": item["id"], "error": str(e)})
            continue

        predicted_cls = chat_resp["classification"]["classification"]
        expected_cls = item["expected_classification"]
        cls_correct = predicted_cls == expected_cls
        if cls_correct:
            classification_correct += 1

        # Retrieval quality: avg score per source
        retrieval_scores = {}
        for sid, chunks in chat_resp.get("retrieved_chunks", {}).items():
            if chunks:
                retrieval_scores[sid] = sum(c["score"] for c in chunks) / len(chunks)

        # Build retrieved context string for the judge (Truncate to avoid 8k token limit)
        context_str = ""
        for sid, chunks in chat_resp.get("retrieved_chunks", {}).items():
            for c in chunks:
                context_str += f"[{sid}] {c['text']}\n"
        
        # Safely truncate context to ~12000 chars (approx 3000 tokens)
        context_str = context_str[:12000] + "\n...(truncated)" if len(context_str) > 12000 else context_str

        # LLM-as-judge
        time.sleep(2)  # pause longer to avoid hitting per-minute rate limits
        judge_scores = llm_judge(
            question=item["question"],
            answer=chat_resp["answer"],
            ground_truth=item["ground_truth"],
            context=context_str
        )

        result = {
            "id": item["id"],
            "question": item["question"],
            "expected_classification": expected_cls,
            "predicted_classification": predicted_cls,
            "classification_correct": cls_correct,
            "retrieval_scores": retrieval_scores,
            "judge_scores": judge_scores,
            "answer_snippet": chat_resp["answer"][:300],
        }
        results.append(result)
        print(f"  Classification: {predicted_cls} (expected {expected_cls}) — {'✓' if cls_correct else '✗'}")
        print(f"  Judge overall: {judge_scores.get('overall', 'N/A')}/100")

    # Summary
    cls_accuracy = classification_correct / total if total > 0 else 0
    avg_judge = {}
    judge_keys = ["accuracy", "faithfulness", "context_recall", "context_precision", "provenance", "overall"]
    for k in judge_keys:
        vals = [r["judge_scores"].get(k, 0) for r in results if "judge_scores" in r]
        avg_judge[k] = round(sum(vals) / len(vals), 2) if vals else 0

    summary = {
        "total_questions": total,
        "classification_accuracy": round(cls_accuracy, 3),
        "avg_judge_scores": avg_judge,
        "results": results,
    }

    output_path = Path(eval_set_path).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"Classification Accuracy: {cls_accuracy:.1%}")
    for k, v in avg_judge.items():
        print(f"  {k}: {v}/100")
    print(f"Results saved to: {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--eval_set", default="eval/eval_set.json")
    parser.add_argument("--base_url", default="http://localhost:8000")
    args = parser.parse_args()
    run_eval(args.session_id, args.eval_set, args.base_url)
