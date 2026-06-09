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

# ─── RAG Query ────────────────────────────────────────────────────────────────

def rag_query(
    collection_name: str,
    question: str,
    top_k: int = settings.top_k_results,
    doc_filter: str | None = None,
) -> dict:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Args:
        collection_name: Which document collection to search
        question: Natural language question
        top_k: Number of chunks to retrieve
        doc_filter: Optional filename to scope search to one document

    Returns:
        Answer with citations and source chunks
    """
    where = {"filename": doc_filter} if doc_filter else None
    chunks = semantic_search(collection_name, question, top_k=top_k, where=where)

    if not chunks:
        return {
            "answer": "No relevant documents found. Please upload documents to this collection first.",
            "sources": [],
            "chunks_used": 0,
        }

    context = _build_context(chunks)

    system = """You are DocBrain, an expert business document analyst.
Answer questions using ONLY the provided document context.
- Be precise and factual
- Cite sources using [Source N] notation
- If the context doesn't contain enough info, say so clearly
- Never hallucinate facts not in the context
- Format numbers, dates, and proper nouns exactly as they appear"""

    user = f"""DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Provide a clear, well-structured answer with citations."""

    answer = _call_claude(system, user, max_tokens=1500)

    return {
        "answer": answer,
        "sources": [
            {
                "filename": c["metadata"].get("filename"),
                "chunk": c["metadata"].get("chunk_index", 0) + 1,
                "relevance_score": c["score"],
                "excerpt": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            }
            for c in chunks
        ],
        "chunks_used": len(chunks),
    }

# ─── Summarization ────────────────────────────────────────────────────────────

SUMMARY_PROMPTS = {
    "executive": """Write a 3-5 sentence executive summary covering:
the document's purpose, key findings or decisions, and recommended actions.
Use clear business language. No jargon.""",

    "detailed": """Write a comprehensive summary covering all major sections.
Use headers (##) to organize. Include key facts, figures, dates, and names.
Aim for 300-500 words.""",

    "bullets": """Summarize as 8-12 bullet points.
Each bullet = one distinct, actionable or factual insight.
Start each with a strong verb or key noun. No fluff.""",

    "risks": """Identify and summarize ONLY risks, issues, deadlines, obligations, or red flags.
Format as a numbered list. For each: state the risk, its potential impact, and location in document.
If no risks found, say so explicitly.""",
}


def summarize_document(
    collection_name: str,
    filename: str,
    style: str = "executive",
) -> dict:
    """
    Generate a structured summary of a specific document.

    Retrieves representative chunks from the doc and summarizes them.
    """
    if style not in SUMMARY_PROMPTS:
        raise ValueError(f"Style must be one of: {list(SUMMARY_PROMPTS.keys())}")

    # Retrieve broad coverage of the document
    chunks = semantic_search(
        collection_name,
        query="main topics key information summary overview",
        top_k=8,
        where={"filename": filename},
    )

    if not chunks:
        raise ValueError(f"Document '{filename}' not found in collection '{collection_name}'")

    context = _build_context(chunks)
    prompt_instruction = SUMMARY_PROMPTS[style]

    system = f"""You are DocBrain, an expert business document analyst.
{prompt_instruction}
Base your summary ONLY on the provided document content."""

    user = f"""DOCUMENT: {filename}

CONTENT:
{context}

Generate the requested summary."""

    summary = _call_claude(system, user, max_tokens=1000)

    return {
        "filename": filename,
        "summary_style": style,
        "summary": summary,
        "based_on_chunks": len(chunks),
    }
# ─── Key Insights Extraction ──────────────────────────────────────────────────

def extract_insights(collection_name: str, filename: str) -> dict:
    """
    Auto-extract structured business intelligence from a document:
    action items, key dates, people/orgs, financial figures, decisions.
    """
    chunks = semantic_search(
        collection_name,
        query="dates deadlines people organizations money actions decisions",
        top_k=10,
        where={"filename": filename},
    )

    if not chunks:
        raise ValueError(f"Document '{filename}' not found in collection '{collection_name}'")

    context = _build_context(chunks)

    system = """You are DocBrain, a business intelligence extractor.
Extract structured data from the document. Return ONLY valid JSON, no markdown fences.
Use this exact schema:
{
  "action_items": [{"task": "...", "owner": "...", "deadline": "..."}],
  "key_dates": [{"date": "...", "event": "..."}],
  "people_and_orgs": [{"name": "...", "role": "..."}],
  "financial_figures": [{"amount": "...", "context": "..."}],
  "key_decisions": ["..."],
  "document_type": "contract|invoice|report|policy|memo|other"
}
If a field has no data, use an empty array. Be precise, only include what's in the text."""

    user = f"""DOCUMENT: {filename}

CONTENT:
{context}

Extract all business intelligence as JSON."""

    import json
    raw = _call_claude(system, user, max_tokens=1500)

    try:
        insights = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON block
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        insights = json.loads(match.group()) if match else {"raw": raw}

    return {
        "filename": filename,
        "insights": insights,
        "chunks_analyzed": len(chunks),
    }

# ─── Cross-Collection Search ──────────────────────────────────────────────────

def compare_documents(
    collection_name: str,
    doc_a: str,
    doc_b: str,
    aspect: str = "key differences and similarities",
) -> dict:
    """Compare two documents on a specific aspect."""
    chunks_a = semantic_search(collection_name, aspect, top_k=4, where={"filename": doc_a})
    chunks_b = semantic_search(collection_name, aspect, top_k=4, where={"filename": doc_b})

    context_a = _build_context(chunks_a) if chunks_a else "No content found."
    context_b = _build_context(chunks_b) if chunks_b else "No content found."

    system = """You are DocBrain, a business document analyst.
Compare two documents on the requested aspect.
Structure your response with: ## Similarities, ## Key Differences, ## Recommendation
Be specific — cite actual content from each document."""

    user = f"""ASPECT TO COMPARE: {aspect}

DOCUMENT A ({doc_a}):
{context_a}

DOCUMENT B ({doc_b}):
{context_b}

Provide a structured comparison."""

    comparison = _call_claude(system, user, max_tokens=1500)

    return {
        "document_a": doc_a,
        "document_b": doc_b,
        "aspect": aspect,
        "comparison": comparison,
    }


