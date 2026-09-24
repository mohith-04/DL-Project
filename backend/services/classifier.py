"""
Groq LLM Call #1 — Cross-source relationship classification.
Classifies how sources relate: agree | partial | conflict
"""
import json
from groq import Groq
from config import get_settings

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
    Returns the parsed JSON classification dict.
    """
    client = Groq(api_key=settings.groq_api_key)
    model = model or settings.llm_model_primary

    # Build context: one section per source
    context_parts = []
    for source_id, chunks in per_source_chunks.items():
        if not chunks:
            continue
        combined = "\n---\n".join(c["text"] for c in chunks)
        context_parts.append(f"=== {source_id} ===\n{combined}")

    if not context_parts:
        return {
            "classification": "agree",
            "confidence": 0.5,
            "reasoning": "No source content available",
            "per_source_summary": {},
            "key_differences": "None",
        }

    context = "\n\n".join(context_parts)
    user_msg = f"""Question: {question}

Source excerpts:
{context}

Analyze how these sources relate to answering this question and return the JSON classification."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as exc:
        print(f"[classifier] Error with {model}: {exc}")
        if model == settings.llm_model_primary:
            # Fallback to smaller model
            return classify_sources(question, per_source_chunks, settings.llm_model_fallback)
        raise
