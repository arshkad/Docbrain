# 🧠 DocBrain — Universal Business Document Intelligence API

A production-grade **RAG (Retrieval-Augmented Generation)** API that lets any business upload documents and instantly query them with natural language. Built with **FastAPI**, **ChromaDB**, and **Claude**.

## Why This Exists

Every business has the same problem: critical knowledge locked in PDFs, contracts, reports, and SOPs that nobody can find or query quickly. DocBrain solves this with a clean REST API any team can integrate in minutes.

**Use cases across industries:**
| Industry | Use Case |
|----------|----------|
| Legal | Query contracts, extract obligations and deadlines |
| Finance | Summarize quarterly reports, extract financial figures |
| HR | Search policy manuals, answer employee questions |
| Healthcare | Parse clinical SOPs, extract protocols |
| Sales | Query RFPs, compare vendor proposals |
| Real Estate | Analyze lease agreements, flag risk clauses |

---

## Architecture

```
Client Request
     │
     ▼
FastAPI (app/main.py)
     │
     ├── /collections  ──► Collections Router
     ├── /documents    ──► Ingestion Pipeline
     │                         ├── Text Extraction (PDF/TXT/MD/CSV)
     │                         ├── Sentence-aware Chunking
     │                         └── ChromaDB Vector Storage
     └── /query        ──► LLM Layer (app/llm.py)
                               ├── Semantic Search (ChromaDB)
                               └── Claude Generation (Anthropic API)
```

**RAG Pipeline:**
1. **Ingest** — documents are parsed, chunked (800 tokens, 150 overlap), embedded via sentence-transformers, stored in ChromaDB
2. **Retrieve** — incoming questions trigger cosine-similarity search across stored chunks
3. **Generate** — top-k chunks passed as context to Claude with grounding instructions
4. **Cite** — response includes source citations with relevance scores
