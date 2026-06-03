"""
LLM (Claude) integration layer.
All prompts and generation logic lives here — easy to swap models.
"""

import anthropic
from app.config import settings
from app.database import semantic_search

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("filename", "unknown")
        page_info = f" (chunk {meta.get('chunk_index', '?')+1}/{meta.get('total_chunks', '?')})"
        parts.append(
            f"[Source {i}: {source}{page_info} | relevance: {chunk['score']:.2f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _call_claude(system: str, user: str, max_tokens: int = 1024) -> str:
    """Single-turn Claude call with error handling."""
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()