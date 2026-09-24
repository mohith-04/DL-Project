"""
Groq LLM Call #2 — Final answer synthesis conditioned on classification result.
Produces an answer with inline provenance tags [Source X].
"""
from groq import Groq
from config import get_settings

settings = get_settings()


SYNTHESIS_SYSTEM = """You are an expert AI tutor. Your job is to give clear, accurate, well-structured answers
that explicitly cite which sources support each claim.

Inline provenance format: use [Source A], [Source B], etc. immediately after any claim derived from that source.
You may use [Source A, Source B] when multiple sources agree on a point.

Formatting guidelines:
- Use markdown: headers (##), bullet points, bold for key terms
- Be educational and thorough, not just a summary
- Maintain a pedagogical tone appropriate for a student learning the topic
"""

AGREE_TEMPLATE = """
The sources agree on this topic. Synthesize a clear, unified explanation.

Classification result: {classification_json}

Sources content:
{context}

Question: {question}

Provide a well-structured answer citing sources inline. Since sources agree, you can cite them together
(e.g., [Source A, Source B]) where they converge.
"""

DIFFER_TEMPLATE = """
The sources {classification} on this topic. Your answer MUST:
1. First, present each source's perspective with inline citations
2. Then provide a reconciled/unified explanation
3. Note which framing is more standard/correct if determinable, or state both are valid

Classification result: {classification_json}

Sources content:
{context}

Question: {question}

Structure your answer with:
## Individual Source Perspectives
(What each source says, with [Source X] tags)

## Reconciled Explanation
(Unified answer noting which is more standard or explaining why both are valid)

## Key Takeaway
(One-paragraph summary for the student)
"""


def synthesize_answer(
    question: str,
    per_source_chunks: dict,
    classification: dict,
    model: str | None = None,
) -> str:
    """
    Generate the final synthesized answer.
    Returns markdown string with inline [Source X] provenance tags.
    """
    client = Groq(api_key=settings.groq_api_key)
    model = model or settings.llm_model_primary

    # Build context
    context_parts = []
    for source_id, chunks in per_source_chunks.items():
        if not chunks:
            continue
        combined = "\n---\n".join(c["text"] for c in chunks)
        context_parts.append(f"=== {source_id} ===\n{combined}")
    context = "\n\n".join(context_parts)

    import json
    classification_json = json.dumps(classification, indent=2)
    cls = classification.get("classification", "agree")

    if cls == "agree":
        user_msg = AGREE_TEMPLATE.format(
            classification_json=classification_json,
            context=context,
            question=question,
        )
    else:
        user_msg = DIFFER_TEMPLATE.format(
            classification=cls,
            classification_json=classification_json,
            context=context,
            question=question,
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[synthesizer] Error with {model}: {exc}")
        if model == settings.llm_model_primary:
            return synthesize_answer(question, per_source_chunks, classification, settings.llm_model_fallback)
        raise
