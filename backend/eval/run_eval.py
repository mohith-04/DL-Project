"""
Comprehensive Evaluation Pipeline for the Multi-Source AI Tutor (100-Question Dataset).

Features:
  1. Batch Pacing: Runs N questions (default: 5) then cools down for M seconds (default: 60s)
     to ensure Groq's rolling 60-second Tokens-Per-Minute (TPM) limit resets to zero.
  2. Multi-Key Pool & Self-Healing: Catches 429 rate limits, waits out cooldowns, and rotates keys.
  3. Context Optimization: Automatically trims context length to save ~60% tokens per question.
  4. Incremental Checkpointing & Resume: Saves after every question; safely resume anytime.
  5. Multi-session mapping & Category filtering.
  6. LaTeX & Markdown research table export.

Usage:
    # Standard run with 5-question batches and 60s reset pauses:
    python eval/run_eval.py --session_id <sid>

    # Custom batching (e.g. 10 questions, 45s pause):
    python eval/run_eval.py --session_id <sid> --batch_size 10 --batch_pause 45

    # Resume an interrupted run:
    python eval/run_eval.py --session_id <sid> --resume
"""
import argparse
import json
import sys
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import httpx
from config import get_settings
from eval.eval_metrics import (
    compute_rouge,
    compute_bleu,
    compute_bertscore,
    compute_classification_metrics,
    compute_stats,
    llm_judge,
    aggregate_judge_scores,
    breakdown_by_field,
    generate_latex_tables,
    generate_markdown_tables,
    avg,
)

settings = get_settings()


def countdown_pause(seconds: float, reason: str = "Batch cooldown"):
    """Display a live countdown pause."""
    total_sec = int(seconds)
    print(f"\n⏸ {reason}: Waiting {total_sec}s for Groq rolling quota reset...")
    for remaining in range(total_sec, 0, -5):
        print(f"   ⏳ Resuming in {remaining}s...", end="\r", flush=True)
        time.sleep(min(5, remaining))
    print(f"   ✅ Cooldown complete. Resuming evaluation!          \n")


def run_eval(
    session_id: Optional[str] = None,
    session_map: Optional[Dict[str, str]] = None,
    category: Optional[str] = None,
    eval_set_path: str = "eval/eval_set.json",
    base_url: str = "http://localhost:8000",
    skip_bertscore: bool = False,
    n_questions: int = None,
    resume: bool = False,
    pause_seconds: float = 2.0,
    batch_size: int = 5,
    batch_pause: float = 60.0,
    eval_model: Optional[str] = None,
):
    eval_file = Path(eval_set_path)
    if not eval_file.exists():
        for candidate in [
            Path(__file__).resolve().parent / eval_file.name,
            Path(__file__).resolve().parent / eval_set_path,
            Path(__file__).resolve().parent.parent / eval_set_path,
        ]:
            if candidate.exists():
                eval_file = candidate
                break

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    # Filter by category if specified
    if category:
        category_clean = category.strip()
        eval_set = [it for it in eval_set if it.get("category", "").lower() == category_clean.lower()]
        if not eval_set:
            valid_cats = sorted(set(it.get("category", "") for it in json.load(open(eval_file, "r", encoding="utf-8"))))
            raise ValueError(f"Category '{category}' not found in eval set. Valid categories are: {valid_cats}")
        print(f"[eval] Filtered to category '{category}': {len(eval_set)} questions.")

    if n_questions:
        eval_set = eval_set[:n_questions]

    total = len(eval_set)
    output_dir = eval_file.parent
    output_path = output_dir / "eval_results.json"
    checkpoint_path = output_dir / "eval_results_checkpoint.json"

    # Handle resume
    existing_results = []
    completed_ids = set()
    if resume:
        resume_file = checkpoint_path if checkpoint_path.exists() else output_path
        if resume_file.exists():
            try:
                with open(resume_file, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                    existing_results = prev_data.get("results", [])
                    completed_ids = {r["id"] for r in existing_results if "error" not in r}
                    print(f"[eval] Resuming from checkpoint ({resume_file.name}): {len(completed_ids)}/{total} questions already completed.")
            except Exception as e:
                print(f"[eval] Could not read existing results to resume: {e}")

    print(f"\n{'='*70}")
    print(f"Multi-Source AI Tutor — Research-Grade Evaluation Pipeline")
    if session_id:
        print(f"Single Session ID: {session_id}")
    if session_map:
        print(f"Session Map:       {json.dumps(session_map, indent=2)}")
    if category:
        print(f"Target Category:   {category}")
    print(f"Total Questions:   {total}")
    print(f"Already Completed: {len(completed_ids)}")
    print(f"Batch Pacing:      {batch_size} questions -> {batch_pause}s cooldown pause")
    print(f"Eval Model:        {eval_model or settings.llm_model_fallback}")
    print(f"Output Directory:  {output_dir}")
    print(f"{'='*70}\n")

    results = list(existing_results)
    batch_counter = 0

    # ── Main Evaluation Loop ──────────────────────────────────────────────────
    for i, item in enumerate(eval_set):
        qid = item["id"]
        if qid in completed_ids:
            continue

        cat = item.get("category", "General")
        diff = item.get("difficulty", "medium")

        # Determine which session_id to use for this question
        active_sid = None
        if session_map and cat in session_map:
            active_sid = session_map[cat]
        elif session_id:
            active_sid = session_id
        elif session_map:
            active_sid = list(session_map.values())[0]

        if not active_sid:
            raise ValueError(f"No session_id found for question {qid} (Category: {cat}). Please supply --session_id or --session_map.")

        print(f"[{i+1}/{total}] [{cat}|{diff}] {qid} — {item['question'][:60]}... (Session: {active_sid[:8]}...)")

        chat_resp = None
        # Self-healing retry loop
        for attempt in range(5):
            try:
                resp = httpx.post(
                    f"{base_url}/api/chat",
                    json={"session_id": active_sid, "question": item["question"]},
                    timeout=180,
                )
                if resp.status_code == 429:
                    print(f"  [rate-limit] Received 429. Pausing 45s for quota reset...")
                    time.sleep(45)
                    continue

                resp.raise_for_status()
                chat_resp = resp.json()
                break
            except Exception as e:
                err_text = str(e).lower()
                is_rate = "429" in err_text or "rate limit" in err_text or "quota" in err_text
                sleep_time = 45 if is_rate else 5 * (attempt + 1)
                print(f"  [retry {attempt+1}/5] {e} (sleeping {sleep_time}s)...")
                time.sleep(sleep_time)
                if attempt == 4:
                    print(f"  ❌ Backend chat call failed after 5 attempts: {e}")
                    results.append({
                        "id": qid,
                        "category": cat,
                        "difficulty": diff,
                        "question": item["question"],
                        "session_id": active_sid,
                        "error": str(e),
                    })

        if not chat_resp:
            continue

        predicted_cls = chat_resp["classification"]["classification"]
        expected_cls = item["expected_classification"]
        cls_correct = (predicted_cls == expected_cls)

        # Retrieval quality calculation
        retrieval_scores = {}
        for sid, chunks in chat_resp.get("retrieved_chunks", {}).items():
            if chunks:
                retrieval_scores[sid] = round(
                    sum(c["score"] for c in chunks) / len(chunks), 4
                )
        avg_retrieval = avg(list(retrieval_scores.values()))

        # Build retrieved context string (compact max 2500 chars to save tokens)
        context_str = ""
        for sid, chunks in chat_resp.get("retrieved_chunks", {}).items():
            for c in chunks:
                context_str += f"[{sid}] {c['text']}\n"
        context_str = context_str[:2500] + "\n...(truncated)" if len(context_str) > 2500 else context_str

        answer = chat_resp["answer"]
        ground_truth = item["ground_truth"]

        # Lexical metrics
        rouge = compute_rouge(answer, ground_truth)
        bleu = compute_bleu(answer, ground_truth)

        # Short inter-call pause
        time.sleep(pause_seconds)

        # LLM-as-Judge (automatic backoff and retry built-in)
        judge = llm_judge(
            question=item["question"],
            answer=answer,
            ground_truth=ground_truth,
            context=context_str,
            model=eval_model or settings.llm_model_fallback,
        )

        result_item = {
            "id": qid,
            "category": cat,
            "difficulty": diff,
            "question": item["question"],
            "session_id": active_sid,
            "expected_classification": expected_cls,
            "predicted_classification": predicted_cls,
            "classification_correct": cls_correct,
            "retrieval_scores": retrieval_scores,
            "avg_retrieval_score": avg_retrieval,
            "rouge": rouge,
            "bleu": bleu,
            "judge_scores": judge,
            "answer_snippet": answer[:350],
        }
        results.append(result_item)
        completed_ids.add(qid)
        batch_counter += 1

        cls_symbol = "[MATCH]" if cls_correct else "[DIFF]"
        print(f"  Classification: {predicted_cls} (expected: {expected_cls}) [{cls_symbol}]")
        print(f"  ROUGE-L: {rouge.get('rougeL', 0):.3f} | BLEU: {bleu:.3f} | Judge Overall: {judge.get('overall', 'N/A')}/100")

        # Save checkpoint after every question
        checkpoint_data = {
            "completed": len(completed_ids),
            "total": total,
            "results": results,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as cf:
            json.dump(checkpoint_data, cf, indent=2)

        # ── Batch Pacing Pause ────────────────────────────────────────────────
        # If batch_size reached and there are still questions left to evaluate
        remaining_questions = total - len(completed_ids)
        if batch_counter >= batch_size and remaining_questions > 0:
            batch_counter = 0
            countdown_pause(
                batch_pause,
                reason=f"Batch of {batch_size} questions finished ({remaining_questions} remaining)",
            )

    # ── Final Aggregation ─────────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]

    # Classification metrics
    y_true = [r["expected_classification"] for r in valid]
    y_pred = [r["predicted_classification"] for r in valid]
    cls_metrics = compute_classification_metrics(y_true, y_pred, labels=["agree", "partial", "conflict"])

    # Lexical metrics with sample stats
    rouge1_vals = [r["rouge"]["rouge1"] for r in valid]
    rouge2_vals = [r["rouge"]["rouge2"] for r in valid]
    rougeL_vals = [r["rouge"]["rougeL"] for r in valid]
    bleu_vals = [r["bleu"] for r in valid]

    lexical_metrics = {
        "rouge1_stats": compute_stats(rouge1_vals),
        "rouge2_stats": compute_stats(rouge2_vals),
        "rougeL_stats": compute_stats(rougeL_vals),
        "bleu_stats": compute_stats(bleu_vals),
    }

    # Retrieval cosine similarity stats
    ret_vals = [r["avg_retrieval_score"] for r in valid if r["avg_retrieval_score"] > 0]
    retrieval_metrics = {
        "avg_cosine_similarity": avg(ret_vals),
        "cosine_similarity_stats": compute_stats(ret_vals),
    }

    # Optional BERTScore
    bertscore_agg = {}
    if not skip_bertscore and valid:
        print("\n[eval] Computing BERTScore over predictions...")
        try:
            id_to_gt = {it["id"]: it["ground_truth"] for it in eval_set}
            predictions_ordered = [r["answer_snippet"] for r in valid]
            references_ordered = [id_to_gt.get(r["id"], "") for r in valid]
            bertscore_agg = compute_bertscore(predictions_ordered, references_ordered)
        except Exception as e:
            print(f"  [eval] BERTScore skipped: {e}")

    # LLM Judge aggregates
    judge_agg = aggregate_judge_scores(valid)

    # Breakdowns
    by_category = breakdown_by_field(valid, "category")
    by_difficulty = breakdown_by_field(valid, "difficulty")

    summary = {
        "session_id": session_id,
        "session_map": session_map,
        "category_filter": category,
        "total_questions": total,
        "valid_results": len(valid),
        "errors": total - len(valid),
        "classification": cls_metrics,
        "lexical": lexical_metrics,
        "semantic": bertscore_agg,
        "retrieval": retrieval_metrics,
        "llm_judge": judge_agg,
        "breakdown_by_category": by_category,
        "breakdown_by_difficulty": by_difficulty,
        "results": results,
    }

    # Save final JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Export LaTeX and Markdown tables
    latex_content = generate_latex_tables(summary)
    latex_path = output_dir / "eval_paper_tables.tex"
    with open(latex_path, "w", encoding="utf-8") as lf:
        lf.write(latex_content)

    markdown_content = generate_markdown_tables(summary)
    markdown_path = output_dir / "eval_paper_tables.md"
    with open(markdown_path, "w", encoding="utf-8") as mf:
        mf.write(markdown_content)

    # Clean up temporary checkpoint file
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    # ── Console Report ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE — {len(valid)}/{total} Questions Evaluated")
    print(f"{'='*70}")
    print(f"\n📊 1. Cross-Source Classification Metrics:")
    print(f"   Accuracy:        {cls_metrics['accuracy']:.1%} [95% CI: {cls_metrics['accuracy_ci95_low']:.1%} - {cls_metrics['accuracy_ci95_high']:.1%}]")
    print(f"   Macro Precision: {cls_metrics['macro_precision']:.3f}")
    print(f"   Macro Recall:    {cls_metrics['macro_recall']:.3f}")
    print(f"   Macro F1:        {cls_metrics['macro_f1']:.3f}")
    print(f"   Per-Class Breakdown:")
    for label, m in cls_metrics.get("per_class", {}).items():
        print(f"     • {label.upper():8s}: Precision={m['precision']:.3f} | Recall={m['recall']:.3f} | F1={m['f1']:.3f} (N={m['support']})")

    print(f"\n📝 2. Lexical & Generation Quality Metrics (Mean ± Std):")
    print(f"   ROUGE-1: {lexical_metrics['rouge1_stats']['mean']:.4f} ± {lexical_metrics['rouge1_stats']['std']:.4f}")
    print(f"   ROUGE-2: {lexical_metrics['rouge2_stats']['mean']:.4f} ± {lexical_metrics['rouge2_stats']['std']:.4f}")
    print(f"   ROUGE-L: {lexical_metrics['rougeL_stats']['mean']:.4f} ± {lexical_metrics['rougeL_stats']['std']:.4f}")
    print(f"   BLEU:    {lexical_metrics['bleu_stats']['mean']:.4f} ± {lexical_metrics['bleu_stats']['std']:.4f}")

    if bertscore_agg:
        print(f"\n🧠 3. Semantic BERTScore:")
        print(f"   F1:        {bertscore_agg.get('bertscore_f1', 0):.4f}")
        print(f"   Precision: {bertscore_agg.get('bertscore_precision', 0):.4f}")
        print(f"   Recall:    {bertscore_agg.get('bertscore_recall', 0):.4f}")

    print(f"\n🔍 4. Retrieval Quality:")
    print(f"   Avg Cosine Similarity: {retrieval_metrics['avg_cosine_similarity']:.4f}")

    print(f"\n⚖️  5. LLM-as-Judge Dimensions (Mean ± Std / 100):")
    for dim in ["accuracy", "faithfulness", "context_recall", "context_precision", "provenance", "overall"]:
        st = judge_agg.get(dim, {})
        print(f"   {dim.replace('_', ' ').title():20s}: {st.get('mean', 0):.1f} ± {st.get('std', 0):.1f}")

    print(f"\n📂 6. Category Breakdown:")
    for cat_name, info in sorted(by_category.items()):
        print(f"   • {cat_name.replace('_', ' '):26s} (N={info['count']}): Acc={info['classification_accuracy']:.1%} | ROUGE-L={info['rouge_l_stats']['mean']:.3f} | Overall={info['overall_stats']['mean']:.1f}/100")

    print(f"\n📈 7. Difficulty Breakdown:")
    for diff in ["easy", "medium", "hard"]:
        if diff in by_difficulty:
            info = by_difficulty[diff]
            print(f"   • {diff.capitalize():10s} (N={info['count']}): Acc={info['classification_accuracy']:.1%} | ROUGE-L={info['rouge_l_stats']['mean']:.3f} | Overall={info['overall_stats']['mean']:.1f}/100")

    print(f"\n💾 Saved Research Artifacts:")
    print(f"   • Raw Results:     {output_path}")
    print(f"   • LaTeX Tables:    {latex_path}")
    print(f"   • Markdown Tables: {markdown_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Source AI Tutor 100-Question Evaluation Pipeline")
    parser.add_argument("--session_id", default=None, help="Single session ID to use for all questions")
    parser.add_argument("--session_map", default=None, help="Path to JSON file or JSON string mapping category names to session IDs")
    parser.add_argument("--category", default=None, help="Filter evaluation to a specific category (e.g. RAG, ML, DL, Transformers, MultiSource_Conflict)")
    parser.add_argument("--eval_set", default="eval/eval_set.json", help="Path to eval set JSON")
    parser.add_argument("--base_url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--skip_bertscore", action="store_true", help="Skip BERTScore calculation")
    parser.add_argument("--n_questions", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress in eval_results.json")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause in seconds between questions within a batch")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of questions per batch before rate limit cooldown pause")
    parser.add_argument("--batch_pause", type=float, default=60.0, help="Cooldown duration in seconds between batches to reset Groq 60s quota")
    parser.add_argument("--eval_model", default=None, help="Specific model for judge and evaluation")
    args = parser.parse_args()

    # Parse session_map if provided
    s_map = None
    if args.session_map:
        if os.path.exists(args.session_map):
            with open(args.session_map, "r", encoding="utf-8") as f:
                s_map = json.load(f)
        else:
            candidates = [
                Path(__file__).resolve().parent / Path(args.session_map).name,
                Path(__file__).resolve().parent / args.session_map,
                Path(__file__).resolve().parent.parent / args.session_map,
            ]
            found = False
            for c in candidates:
                if c.exists():
                    with open(c, "r", encoding="utf-8") as f:
                        s_map = json.load(f)
                    found = True
                    break
            if not found:
                try:
                    s_map = json.loads(args.session_map)
                except Exception as e:
                    raise ValueError(f"Could not parse --session_map as JSON or file: {e}")

    if not args.session_id and not s_map:
        parser.error("You must provide either --session_id or --session_map (or both).")

    run_eval(
        session_id=args.session_id,
        session_map=s_map,
        category=args.category,
        eval_set_path=args.eval_set,
        base_url=args.base_url,
        skip_bertscore=args.skip_bertscore,
        n_questions=args.n_questions,
        resume=args.resume,
        pause_seconds=args.pause,
        batch_size=args.batch_size,
        batch_pause=args.batch_pause,
        eval_model=args.eval_model,
    )
