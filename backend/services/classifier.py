"""
Groq LLM Call #1 — Cross-source relationship classification.
Classifies how sources relate: agree | partial | conflict
Enhanced with Multi-Key Pool, Smart 429 Backoff, and Token Optimization.
"""
import json
from config import get_settings
from services.llm_client import safe_chat_completion, build_compact_context

settings = get_settings()

CLASSIFY_SYSTEM = """You are an expert knowledge analyst specializing in comparing educational sources.
Your task: analyze what multiple sources say about a specific question and classify their relationship.

Respond ONLY with a valid JSON object in this exact format:
{
  "classification": "agree" | "partial" | "conflict",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why you chose this classification",
  "per_source_summary": {
    "Source A": "One sentence summary of what Source A says",
    "Source B": "One sentence summary of what Source B says",
    ... (include all sources present)
  },
  "key_differences": "Describe the main differences, or 'None' if sources fully agree"
}

Classification definitions:
- agree: All sources convey essentially the same concept/answer
- partial: Sources cover the same topic but differ in emphasis, terminology, depth, or framing
- conflict: Sources make factually contradictory claims or recommend opposite approaches"""


def classify_sources(
    question: str,
    per_source_chunks: dict,
    model: str | None = None,
) -> dict:
    """
    Given per-source retrieved chunks, classify their relationship for the question.
    Uses token-optimized context and safe_chat_completion with backoff/rotation.
    """
    context = build_compact_context(per_source_chunks, max_chars_per_source=1000)

    if not context:
        return {
            "classification": "agree",
            "confidence": 0.5,
            "reasoning": "No source content available",
            "per_source_summary": {},
            "key_differences": "None",
        }

    user_msg = f"""Question: {question}

Source excerpts:
{context}

Analyze how these sources relate to answering this question and return the JSON classification."""

    target_model = model or settings.llm_model_primary

    try:
        response = safe_chat_completion(
            model=target_model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as exc:
        print(f"[classifier] Exception encountered: {exc}")
        if target_model != settings.llm_model_fallback and settings.llm_model_fallback:
            return classify_sources(question, per_source_chunks, settings.llm_model_fallback)
        # Safe fallback structure if all retries exhausted
        return {
            "classification": "agree",
            "confidence": 0.5,
            "reasoning": f"Classification fallback due to error: {exc}",
            "per_source_summary": {},
            "key_differences": "None",
        }
