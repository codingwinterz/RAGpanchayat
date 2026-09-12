# Constitution of India RAG Chatbot — Project Plan

A domain-locked Retrieval-Augmented Generation chatbot that answers questions strictly from the Constitution of India, with cited article numbers and an "I don't know" fallback when the answer isn't in scope.

## Why This Project

- Public, clean, free source document — zero scraping or licensing issues.
- Articles are numbered and self-contained, giving high-precision retrieval that's easy to prove correct.
- Broad appeal — non-technical judges immediately understand the value.
- Natural stretch goal (Hindi + English support) fits an Indian hackathon context well.
- Built from scratch, no dependency on prior projects.

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Backend API | FastAPI | REST endpoints, async, auto-docs via `/docs` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, local, no API key |
| Vector store | ChromaDB | Local, persistent, zero infra cost |
| Generation | Gemini API (free tier) | Already used in a past project, generous free quota |
| Demo UI | Streamlit or FastAPI `/docs` | Fast to stand up, no frontend work needed |
| IDE | Antigravity | Agentic scaffolding for boilerplate |
| Containerization (stretch) | Docker | Deployment-ready story |

## Phase 1 — Data Prep (Day 1, few hours)

- Download the official Constitution of India PDF (Ministry of Law & Justice site).
- Extract text (e.g., `pdfplumber` or `PyMuPDF`).
- Split into chunks by Article number using regex on headers like `Article \d+` — semantically complete units beat fixed-size chunking here.
- Attach metadata to each chunk: `{article_number, part, title}`.

## Phase 2 — Embedding + Vector Store (Day 1)

- Embed each chunk with `all-MiniLM-L6-v2`.
- Store in ChromaDB with metadata attached.
- ~450 articles → indexing takes seconds, no infra needed.

## Phase 3 — Retrieval + Generation (Day 2)

- Build a `/ask` FastAPI endpoint:
  1. Embed incoming question.
  2. Top-k (3–5) similarity search against Chroma.
  3. Pass retrieved chunks + question into a Gemini prompt.
- System prompt constraint: *"Answer only using the provided context. Cite the article number. If the answer isn't in the context, say you don't know."*
- Response shape: `{answer, cited_articles: [...]}`

## Phase 4 — Guardrails (Day 2)

- Reject or flag out-of-scope questions (e.g., "what's the weather") with a fixed response: "I only answer questions about the Indian Constitution."
- Log retrieval similarity scores. If the top match is below a set threshold, trigger the "I don't know" fallback instead of guessing.

## Phase 5 — Evaluation (Day 2–3)

- Hand-write 20–25 Q&A pairs covering well-known articles (Article 21 — right to life, Article 19 — freedoms, Article 32 — constitutional remedies, etc.).
- Run them through the pipeline, manually verify correctness.
- Report accuracy % and average retrieval precision — a concrete number most hackathon RAG demos skip entirely.

## Phase 6 — Demo Layer (Day 3)

- Streamlit single-page chat UI, or a clean walkthrough of FastAPI's `/docs` (Swagger UI).
- Pre-test 3–4 questions to run live during judging — don't gamble on cold retrieval in front of judges.

## Phase 7 — Stretch Goals (only if time remains)

- **Hybrid search**: combine BM25 (keyword) + vector search via reciprocal rank fusion — helps with exact legal terminology.
- **Hindi query support**: translate incoming Hindi queries to English before embedding, or switch to a multilingual embedding model.
- **Dockerize**: wrap the FastAPI app in a Dockerfile + docker-compose for a "deployment-ready" story.
- **GitHub**: proper repo with README, commit history, and version control — reinforces the "production practice" narrative.

## Suggested Folder Structure

```
constitution-rag-bot/
├── data/
│   └── constitution.pdf
├── ingestion/
│   ├── extract_text.py       # PDF → raw text
│   └── chunk_by_article.py   # raw text → article-level chunks + metadata
├── embeddings/
│   └── build_index.py        # chunks → embeddings → Chroma index
├── app/
│   ├── main.py                # FastAPI app, /ask endpoint
│   ├── retriever.py           # similarity search + threshold logic
│   └── prompt.py              # system prompt + Gemini call
├── eval/
│   ├── qa_pairs.json          # hand-written 20-25 Q&A set
│   └── run_eval.py            # accuracy + precision report
├── demo/
│   └── streamlit_app.py       # optional chat UI
├── Dockerfile                 # stretch goal
├── docker-compose.yml         # stretch goal
├── requirements.txt
└── README.md
```

## Timeline Summary

| Day | Focus |
|---|---|
| 1 | Data prep + embedding pipeline |
| 2 | Retrieval, generation, guardrails, start eval |
| 3 | Finish eval, demo layer, stretch goals if time allows |
