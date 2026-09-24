# Multi-Source AI Tutor with Cross-Source Synthesis

> **Powered by Llama 3 (70B) via Groq** | FastAPI + Next.js | FAISS + sentence-transformers

---

## Why This Is Different from Standard Single-Source RAG

A typical RAG chatbot merges all retrieved chunks into a single context window and asks the LLM to "answer based on the following." This has a critical flaw: **it silently hides disagreements between sources**.

This project explicitly keeps sources separate and runs a two-step LLM process:

| Standard RAG | Multi-Source AI Tutor |
|---|---|
| All chunks merged into one blob | Each source stays a labeled document (Source A/B/C/D) |
| Single LLM call | **Two sequential LLM calls**: classify → synthesize |
| No conflict awareness | Explicitly detects agree/partial/conflict |
| No provenance | Inline `[Source X]` citations in every answer |
| Top-k globally | **Top-k per source** (no source drowns out another) |

The core novel mechanism: **conflict detection as a first-class step**, not an afterthought.

---

## Architecture

```
User submits topic URL + question
         │
         ▼
┌─────────────────────────────────────────────────┐
│              INGESTION PIPELINE                   │
│  1. DuckDuckGo auto-discovers 2-3 related URLs   │
│  2. trafilatura/BS4 scrapes each URL separately  │
│  3. Text chunked (512 words, 64 overlap)          │
│  4. sentence-transformers embeds each chunk       │
│  5. FAISS index built with source_id metadata    │
└─────────────────────────────────────────────────┘
         │  session_id
         ▼
┌─────────────────────────────────────────────────┐
│             RETRIEVAL PIPELINE                    │
│  top-k chunks retrieved PER SOURCE independently │
│  (not global top-k — ensures fair representation)│
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│         LLM CALL #1 — CLASSIFICATION             │
│  Groq (llama3-70b-8192) compares source excerpts │
│  → agree | partial | conflict                    │
│  → per-source summary + key differences          │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│         LLM CALL #2 — SYNTHESIS                  │
│  Conditioned on classification:                  │
│  • agree → unified answer with joint citations   │
│  • partial/conflict → "Source A says X, B says Y"│
│    + reconciled explanation                      │
│  All claims tagged [Source A] inline             │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│               REACT FRONTEND                      │
│  • Chat panel with provenance-tagged answers     │
│  • Source comparison panel (right sidebar)       │
│  • Classification badge (agree/partial/conflict) │
│  • Expandable per-source summaries + raw chunks  │
└─────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- A free [Groq API key](https://console.groq.com/)

### 1. Backend

```bash
cd ai-tutor/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
copy .env.example .env
# Edit .env and paste your GROQ_API_KEY

# Start the server
python main.py
# → http://localhost:8000
# → Docs: http://localhost:8000/docs
```

### 2. Frontend

```bash
cd ai-tutor/frontend

npm install
npm run dev
# → http://localhost:3000
```

### 3. Usage

1. Open `http://localhost:3000`
2. Enter a **Topic Name** and **Primary Source URL** (e.g., Wikipedia article on Dijkstra's algorithm)
3. Click **Load & Index Sources** — the system will:
   - Auto-discover 2-3 related URLs from different domains
   - Scrape and embed all sources separately
   - Build a FAISS index tagged by source
4. Ask questions in the chat panel
5. View the **Source Comparison Panel** on the right for classification and per-source summaries

---

## API Reference

### `POST /api/ingest`

```json
{
  "topic_url": "https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm",
  "topic_name": "Dijkstra's Algorithm",
  "n_related": 3
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "topic_name": "Dijkstra's Algorithm",
  "sources": [
    {"source_id": "Source A", "url": "...", "title": "...", "char_count": 12345},
    ...
  ],
  "total_chunks": 87
}
```

### `POST /api/chat`

```json
{
  "session_id": "uuid",
  "question": "How does Dijkstra's algorithm handle negative edge weights?"
}
```

**Response:**
```json
{
  "answer": "## Negative Edge Weights\n\nDijkstra's algorithm **cannot** handle negative edge weights [Source A, Source B]...",
  "classification": {
    "classification": "agree",
    "confidence": 0.95,
    "reasoning": "All sources consistently state that Dijkstra fails with negative weights.",
    "key_differences": "None"
  },
  "source_summaries": [...],
  "retrieved_chunks": {"Source A": [...], "Source B": [...]},
  "session_id": "uuid",
  "question": "..."
}
```

---

## Evaluation

```bash
cd ai-tutor/backend

# First, ingest a session (or get session_id from frontend)
# Then run the eval pipeline:
python eval/run_eval.py --session_id <your_session_id> --eval_set eval/eval_set.json
```

The eval pipeline measures:
1. **Retrieval quality** — average cosine similarity score per source
2. **Classification accuracy** — agree/partial/conflict vs. ground truth labels
3. **Answer quality** — LLM-as-judge scoring: groundedness, accuracy, clarity, provenance (1-5 each)

Results saved to `eval/eval_results.json`.

---

## Project Structure

```
ai-tutor/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings (env-based)
│   ├── routers/
│   │   ├── ingest.py        # POST /api/ingest
│   │   └── chat.py          # POST /api/chat
│   ├── services/
│   │   ├── discovery.py     # DuckDuckGo URL discovery
│   │   ├── scraper.py       # trafilatura + BS4 scraping
│   │   ├── embedder.py      # sentence-transformers + FAISS
│   │   ├── retriever.py     # Per-source top-k retrieval
│   │   ├── classifier.py    # Groq LLM call #1 (classify)
│   │   └── synthesizer.py   # Groq LLM call #2 (synthesize)
│   ├── eval/
│   │   ├── eval_set.json    # 5 ground-truth QA pairs
│   │   └── run_eval.py      # Evaluation pipeline
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/             # Next.js App Router pages
│       ├── components/
│       │   ├── IngestPanel.tsx     # URL input + source list
│       │   ├── ChatPanel.tsx       # Chat interface
│       │   ├── AnswerCard.tsx      # Markdown + provenance tags
│       │   ├── SourcePanel.tsx     # Classification + source summaries
│       │   └── ProvenanceTag.tsx   # Colored [Source X] badges
│       └── types/index.ts
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3 70B via [Groq](https://groq.com/) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Store | FAISS (IndexFlatIP, cosine similarity) |
| Web Scraping | trafilatura + BeautifulSoup4 |
| URL Discovery | DuckDuckGo Search API |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 14 + Tailwind CSS |

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (required) |

Never hardcode API keys. Use `.env` (git-ignored).

---

## License

MIT
