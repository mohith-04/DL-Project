"""
Comprehensive evaluation metrics for the Multi-Source AI Tutor.
Computes ROUGE, BLEU, BERTScore, LLM-as-judge, classification metrics,
retrieval quality metrics, statistical confidence intervals (95% CI, StdDev, SE),
and LaTeX / Markdown research paper table generation.
"""
import json
import math
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Statistical Utilities (Research-Grade) ───────────────────────────────────
def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, sample standard deviation, standard error, and 95% CI."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "se": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "count": 0}
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    z = 1.96  # 95% confidence
    ci_low = max(0.0, mean - z * se)
    ci_high = mean + z * se
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "se": round(se, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "count": n,
    }


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Wilson score interval for binomial proportions (accuracy).
    Preferred in academic literature over normal approximation for small or skewed proportions.
    """
    if n == 0:
        return {"accuracy": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    p = k / n
    z = 1.96 if confidence == 0.95 else 1.645
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return {
        "accuracy": round(p, 4),
        "ci95_low": round(max(0.0, center - margin), 4),
        "ci95_high": round(min(1.0, center + margin), 4),
    }


def avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


# ── ROUGE ─────────────────────────────────────────────────────────────────────
def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ── BLEU ──────────────────────────────────────────────────────────────────────
def compute_bleu(prediction: str, reference: str) -> float:
    """Compute sentence BLEU score (smoothed)."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        import nltk
        try:
            ref_tokens = word_tokenize(reference.lower())
            pred_tokens = word_tokenize(prediction.lower())
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            ref_tokens = word_tokenize(reference.lower())
            pred_tokens = word_tokenize(prediction.lower())
        sf = SmoothingFunction().method4
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=sf)
        return round(score, 4)
    except ImportError:
        return 0.0


# ── BERTScore ─────────────────────────────────────────────────────────────────
def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BERTScore P, R, F1 for a batch."""
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            predictions,
            references,
            model_type="distilbert-base-uncased",
            lang="en",
            verbose=False,
        )
        return {
            "bertscore_precision": round(P.mean().item(), 4),
            "bertscore_recall": round(R.mean().item(), 4),
            "bertscore_f1": round(F1.mean().item(), 4),
        }
    except ImportError:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


# ── Classification Metrics ────────────────────────────────────────────────────
def compute_classification_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str] = None
) -> Dict:
    """Compute per-class and macro precision, recall, F1 with Wilson 95% CI."""
    if labels is None:
        labels = sorted(set(y_true + y_pred))

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    per_class = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": y_true.count(label),
        }

    macro_p = sum(v["precision"] for v in per_class.values()) / len(labels) if labels else 0
    macro_r = sum(v["recall"] for v in per_class.values()) / len(labels) if labels else 0
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels) if labels else 0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    wilson = compute_wilson_ci(correct, total)

    return {
        "accuracy": wilson["accuracy"],
        "accuracy_ci95_low": wilson["ci95_low"],
        "accuracy_ci95_high": wilson["ci95_high"],
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert academic evaluator for a Multi-Source Retrieval-Augmented Generation (RAG) tutoring system.
Score the AI tutor's response on the following dimensions using integer scores from 0 to 100:

{
  "accuracy": 0-100,           // Factual accuracy compared to the ground truth
  "faithfulness": 0-100,       // Are all claims supported by retrieved context? (0 = total hallucination, 100 = perfectly grounded)
  "context_recall": 0-100,     // Does retrieved context contain the information required to answer?
  "context_precision": 0-100,  // Is the retrieved context relevant and concise (low noise)?
  "provenance": 0-100,         // Does the answer provide clear citations e.g. [Source A], [Source B]?
  "completeness": 0-100,       // Does the answer address all facets of the inquiry?
  "coherence": 0-100,          // Is the response well-organized, educational, and clear?
  "overall": 0-100,            // Comprehensive score reflecting overall pedagogical and factual quality
  "justification": "Concise summary of rationale"
}

IMPORTANT: Output ONLY a single, strictly valid JSON object. No markdown fences, no trailing commas."""


def llm_judge(
    question: str,
    answer: str,
    ground_truth: str,
    context: str,
    client=None,
    model: Optional[str] = None,
) -> Dict:
    """LLM-as-judge evaluation returning multi-dimensional scores with robust retry and token-optimized context."""
    from services.llm_client import safe_chat_completion
    from config import get_settings
    settings = get_settings()

    # Truncate context to max 2500 chars to avoid token limit overflow
    compact_context = context[:2500] + ("\n...(truncated)" if len(context) > 2500 else "")

    user_msg = (
        f"Question: {question}\n\n"
        f"Ground Truth Reference:\n{ground_truth}\n\n"
        f"Retrieved Context:\n{compact_context}\n\n"
        f"AI Tutor Generated Answer:\n{answer}"
    )

    judge_model = model or settings.llm_model_fallback

    try:
        resp = safe_chat_completion(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        try:
            err_body = e.response.json() if hasattr(e, "response") else {}
            failed_text = err_body.get("error", {}).get("failed_generation", "")
            if failed_text:
                cleaned = re.sub(r"\}\}\s*$", "}", failed_text.strip())
                return json.loads(cleaned)
        except Exception:
            pass
        print(f"[judge] LLM judge fallback: {e}")

    return {
        k: 0
        for k in [
            "accuracy", "faithfulness", "context_recall", "context_precision",
            "provenance", "completeness", "coherence", "overall"
        ]
    }


def aggregate_judge_scores(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute stats (mean, std, se, ci95) for each judge score dimension."""
    judge_keys = [
        "accuracy", "faithfulness", "context_recall", "context_precision",
        "provenance", "completeness", "coherence", "overall"
    ]
    agg = {}
    for k in judge_keys:
        vals = [r["judge_scores"].get(k, 0) for r in results if "judge_scores" in r and r["judge_scores"].get(k, 0) > 0]
        agg[k] = compute_stats(vals)
    return agg


def breakdown_by_field(results: List[Dict], field: str) -> Dict[str, Dict]:
    """Group results by field (category or difficulty) and compute comprehensive metrics."""
    groups = defaultdict(list)
    for r in results:
        if "judge_scores" not in r:
            continue
        key = r.get(field, "unknown")
        groups[key].append(r)

    breakdown = {}
    for key, items in groups.items():
        total_items = len(items)
        correct_cls = sum(1 for it in items if it.get("classification_correct", False))
        cls_acc = round(correct_cls / total_items, 4) if total_items > 0 else 0.0

        rouge_l_vals = [it["rouge"]["rougeL"] for it in items if "rouge" in it and "rougeL" in it["rouge"]]
        judge_overall_vals = [it["judge_scores"].get("overall", 0) for it in items if "judge_scores" in it]
        faith_vals = [it["judge_scores"].get("faithfulness", 0) for it in items if "judge_scores" in it]
        acc_vals = [it["judge_scores"].get("accuracy", 0) for it in items if "judge_scores" in it]

        breakdown[key] = {
            "count": total_items,
            "classification_accuracy": cls_acc,
            "rouge_l_stats": compute_stats(rouge_l_vals),
            "faithfulness_stats": compute_stats(faith_vals),
            "accuracy_stats": compute_stats(acc_vals),
            "overall_stats": compute_stats(judge_overall_vals),
        }
    return breakdown


# ── Paper Table Generators (LaTeX & Markdown) ────────────────────────────────
def generate_latex_tables(summary: Dict) -> str:
    """Generate publication-ready LaTeX tables for insertion into paper drafts."""
    cls_m = summary.get("classification", {})
    lex_m = summary.get("lexical", {})
    judge_m = summary.get("llm_judge", {})
    by_cat = summary.get("breakdown_by_category", {})
    by_diff = summary.get("breakdown_by_difficulty", {})

    tex = []
    tex.append("% ========================================================================")
    tex.append("% TABLE 1: Overall Performance Metrics for Multi-Source AI Tutor")
    tex.append("% ========================================================================")
    tex.append("\\begin{table}[htbp]")
    tex.append("\\centering")
    tex.append("\\caption{Overall Evaluation Metrics Across $N=" + str(summary.get("valid_results", 100)) + "$ Questions}")
    tex.append("\\label{tab:overall_results}")
    tex.append("\\begin{tabular}{llr}")
    tex.append("\\toprule")
    tex.append("\\textbf{Dimension} & \\textbf{Metric} & \\textbf{Score} \\\\")
    tex.append("\\midrule")
    acc_pct = cls_m.get('accuracy', 0) * 100
    ci_low = cls_m.get('accuracy_ci95_low', 0) * 100
    ci_high = cls_m.get('accuracy_ci95_high', 0) * 100
    tex.append(f"Classification & Accuracy & {acc_pct:.1f}\\% [95\\% CI: {ci_low:.1f}\\%--{ci_high:.1f}\\%] \\\\")
    tex.append(f"Classification & Macro Precision & {cls_m.get('macro_precision', 0):.3f} \\\\")
    tex.append(f"Classification & Macro Recall & {cls_m.get('macro_recall', 0):.3f} \\\\")
    tex.append(f"Classification & Macro F1 & {cls_m.get('macro_f1', 0):.3f} \\\\")
    tex.append("\\midrule")
    r1 = lex_m.get("rouge1_stats", {})
    r2 = lex_m.get("rouge2_stats", {})
    rl = lex_m.get("rougeL_stats", {})
    bl = lex_m.get("bleu_stats", {})
    tex.append(f"Lexical Overlap & ROUGE-1 & {r1.get('mean', 0):.3f} $\\pm$ {r1.get('std', 0):.3f} \\\\")
    tex.append(f"Lexical Overlap & ROUGE-2 & {r2.get('mean', 0):.3f} $\\pm$ {r2.get('std', 0):.3f} \\\\")
    tex.append(f"Lexical Overlap & ROUGE-L & {rl.get('mean', 0):.3f} $\\pm$ {rl.get('std', 0):.3f} \\\\")
    tex.append(f"Lexical Overlap & BLEU & {bl.get('mean', 0):.3f} $\\pm$ {bl.get('std', 0):.3f} \\\\")
    tex.append("\\midrule")
    ret_score = summary.get("retrieval", {}).get("avg_cosine_similarity", 0)
    tex.append(f"Retrieval & Cosine Similarity & {ret_score:.4f} \\\\")
    tex.append("\\midrule")
    for dim in ["accuracy", "faithfulness", "context_recall", "context_precision", "provenance", "overall"]:
        st = judge_m.get(dim, {})
        dim_label = dim.replace("_", " ").title()
        tex.append(f"LLM-as-Judge & {dim_label} & {st.get('mean', 0):.1f} $\\pm$ {st.get('std', 0):.1f} / 100 \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}\n")

    tex.append("% ========================================================================")
    tex.append("% TABLE 2: Category Breakdown (5 Categories x 20 Questions)")
    tex.append("% ========================================================================")
    tex.append("\\begin{table}[htbp]")
    tex.append("\\centering")
    tex.append("\\caption{Performance Breakdown across Subject Categories ($N=20$ per category)}")
    tex.append("\\label{tab:category_breakdown}")
    tex.append("\\begin{tabular}{lcccc}")
    tex.append("\\toprule")
    tex.append("\\textbf{Category} & \\textbf{Cls Acc (\\%)} & \\textbf{ROUGE-L} & \\textbf{Faithfulness} & \\textbf{Overall (/100)} \\\\")
    tex.append("\\midrule")
    for cat, info in sorted(by_cat.items()):
        c_acc = f"{info.get('classification_accuracy', 0) * 100:.1f}\\%"
        rl_m = f"{info.get('rouge_l_stats', {}).get('mean', 0):.3f}"
        f_m = f"{info.get('faithfulness_stats', {}).get('mean', 0):.1f}"
        ov_m = f"{info.get('overall_stats', {}).get('mean', 0):.1f}"
        cat_disp = cat.replace("_", " ")
        tex.append(f"{cat_disp} & {c_acc} & {rl_m} & {f_m} & {ov_m} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}\n")

    tex.append("% ========================================================================")
    tex.append("% TABLE 3: Per-Class Classification Confusion Matrix / Precision-Recall-F1")
    tex.append("% ========================================================================")
    tex.append("\\begin{table}[htbp]")
    tex.append("\\centering")
    tex.append("\\caption{Cross-Source Relationship Classification Metrics by Class}")
    tex.append("\\label{tab:per_class_classification}")
    tex.append("\\begin{tabular}{lcccc}")
    tex.append("\\toprule")
    tex.append("\\textbf{Class} & \\textbf{Support ($N$)} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\")
    tex.append("\\midrule")
    for label, m in cls_m.get("per_class", {}).items():
        tex.append(f"{label.capitalize()} & {m.get('support', 0)} & {m.get('precision', 0):.3f} & {m.get('recall', 0):.3f} & {m.get('f1', 0):.3f} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")

    return "\n".join(tex)


def generate_markdown_tables(summary: Dict) -> str:
    """Generate Markdown tables suitable for GitHub, docs, or thesis markdown drafts."""
    cls_m = summary.get("classification", {})
    lex_m = summary.get("lexical", {})
    judge_m = summary.get("llm_judge", {})
    by_cat = summary.get("breakdown_by_category", {})
    by_diff = summary.get("breakdown_by_difficulty", {})

    md = []
    md.append("# Multi-Source AI Tutor — Evaluation Results (N=100 Questions)\n")

    md.append("### Table 1: Overall System Performance Metrics")
    md.append("| Dimension | Metric | Score | 95% Confidence Interval |")
    md.append("|---|---|---|---|")
    md.append(f"| **Classification** | Accuracy | **{cls_m.get('accuracy', 0):.1%}** | [{cls_m.get('accuracy_ci95_low', 0):.1%}, {cls_m.get('accuracy_ci95_high', 0):.1%}] |")
    md.append(f"| Classification | Macro Precision | {cls_m.get('macro_precision', 0):.3f} | - |")
    md.append(f"| Classification | Macro Recall | {cls_m.get('macro_recall', 0):.3f} | - |")
    md.append(f"| Classification | Macro F1 | **{cls_m.get('macro_f1', 0):.3f}** | - |")

    r1 = lex_m.get("rouge1_stats", {})
    r2 = lex_m.get("rouge2_stats", {})
    rl = lex_m.get("rougeL_stats", {})
    bl = lex_m.get("bleu_stats", {})
    md.append(f"| **Lexical** | ROUGE-1 | {r1.get('mean', 0):.4f} ± {r1.get('std', 0):.4f} | [{r1.get('ci95_low', 0):.4f}, {r1.get('ci95_high', 0):.4f}] |")
    md.append(f"| Lexical | ROUGE-2 | {r2.get('mean', 0):.4f} ± {r2.get('std', 0):.4f} | [{r2.get('ci95_low', 0):.4f}, {r2.get('ci95_high', 0):.4f}] |")
    md.append(f"| Lexical | ROUGE-L | **{rl.get('mean', 0):.4f} ± {rl.get('std', 0):.4f}** | [{rl.get('ci95_low', 0):.4f}, {rl.get('ci95_high', 0):.4f}] |")
    md.append(f"| Lexical | BLEU | {bl.get('mean', 0):.4f} ± {bl.get('std', 0):.4f} | [{bl.get('ci95_low', 0):.4f}, {bl.get('ci95_high', 0):.4f}] |")

    ret_score = summary.get("retrieval", {}).get("avg_cosine_similarity", 0)
    md.append(f"| **Retrieval** | Avg Cosine Sim | {ret_score:.4f} | - |")

    for dim in ["accuracy", "faithfulness", "context_recall", "context_precision", "provenance", "overall"]:
        st = judge_m.get(dim, {})
        dim_label = dim.replace("_", " ").title()
        md.append(f"| **LLM-as-Judge** | {dim_label} | **{st.get('mean', 0):.1f} ± {st.get('std', 0):.1f} / 100** | [{st.get('ci95_low', 0):.1f}, {st.get('ci95_high', 0):.1f}] |")

    md.append("\n### Table 2: Performance Breakdown across 5 Categories (20 Questions each)")
    md.append("| Category | Questions | Classification Accuracy | ROUGE-L | Faithfulness | Overall (/100) |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for cat, info in sorted(by_cat.items()):
        cat_disp = cat.replace("_", " ")
        c_acc = f"{info.get('classification_accuracy', 0) * 100:.1f}\\%"
        rl_m = f"{info.get('rouge_l_stats', {}).get('mean', 0):.3f}"
        f_m = f"{info.get('faithfulness_stats', {}).get('mean', 0):.1f}"
        ov_m = f"{info.get('overall_stats', {}).get('mean', 0):.1f}"
        md.append(f"| **{cat_disp}** | {info.get('count', 20)} | {c_acc} | {rl_m} | {f_m} | **{ov_m}** |")

    md.append("\n### Table 3: Performance Breakdown by Difficulty")
    md.append("| Difficulty | Questions | Classification Accuracy | ROUGE-L | Faithfulness | Overall (/100) |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for diff in ["easy", "medium", "hard"]:
        if diff in by_diff:
            info = by_diff[diff]
            c_acc = f"{info.get('classification_accuracy', 0) * 100:.1f}\\%"
            rl_m = f"{info.get('rouge_l_stats', {}).get('mean', 0):.3f}"
            f_m = f"{info.get('faithfulness_stats', {}).get('mean', 0):.1f}"
            ov_m = f"{info.get('overall_stats', {}).get('mean', 0):.1f}"
            md.append(f"| **{diff.capitalize()}** | {info.get('count', 0)} | {c_acc} | {rl_m} | {f_m} | **{ov_m}** |")

    md.append("\n### Table 4: Per-Class Classification (Agree vs Partial vs Conflict)")
    md.append("| Target Class | Support ($N$) | Precision | Recall | F1-Score |")
    md.append("|---|:---:|:---:|:---:|:---:|")
    for label, m in cls_m.get("per_class", {}).items():
        md.append(f"| **{label.capitalize()}** | {m.get('support', 0)} | {m.get('precision', 0):.3f} | {m.get('recall', 0):.3f} | **{m.get('f1', 0):.3f}** |")

    return "\n".join(md)
