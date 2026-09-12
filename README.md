# Constitution of India — Domain-Locked RAG Chatbot

A domain-locked Retrieval-Augmented Generation (RAG) assistant that strictly answers legal queries from the official **Constitution of India**, cites verified Article numbers, and rejects out-of-scope queries using distance guardrails.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    User([User Question]) --> Retriever[app/retriever.py: Embed query via all-MiniLM-L6-v2]
    Retriever --> Chroma[(ChromaDB: Cosine Similarity Search top-k=5)]
    Chroma --> Decision{Top Distance > 0.75?}
    Decision -- Yes --> Fallback[Return fallback message: Skips LLM call completely]
    Decision -- No --> Prompt[app/prompt.py: Grounded Prompt + Gemini 3.6 Flash]
    Prompt --> API[app/main.py: JSON Response with answer, cited_articles, retrieved_sources]
    API --> UI[demo/streamlit_app.py: Interactive Chat UI]
```

---

## 📂 Project Structure

```
.
├── data/
│   ├── constitution.pdf          # Official Constitution of India PDF (diglot edition)
│   ├── constitution_raw.txt      # Raw extracted text from PDF
│   ├── articles_chunked.json     # 409 extracted article chunks with metadata
│   └── overview_chunks.json      # 5 hand-written overview chunks for broad/summary topics
├── ingestion/
│   ├── extract_text.py           # Extracts text from PDF to constitution_raw.txt
│   └── chunk_by_article.py       # Regex chunking into articles (tags main_body vs amendment_act)
├── scripts/
│   ├── check_chunks.py           # Diagnostic verification script for chunk quality
│   └── test_ask.py               # Test script sending queries to the FastAPI /ask endpoint
├── embeddings/
│   └── build_index.py            # Embeds articles + overview chunks into local ChromaDB
├── app/
│   ├── main.py                   # FastAPI backend with /ask and /health endpoints
│   ├── retriever.py              # Semantic similarity retriever with source prioritization
│   └── prompt.py                 # Gemini prompt construction, grounding, & citation parser
├── eval/
│   ├── qa_pairs.json             # 25 benchmark QA pairs across constitutional topics
│   └── run_eval.py               # Evaluation benchmark script measuring Top-1 and Top-k accuracy
├── demo/
│   └── streamlit_app.py          # Streamlit chat interface with citation badges & source inspector
├── requirements.txt              # Project dependencies
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore file for secrets, venv, and vector DB
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone & Environment Setup

Clone the repository and set up a Python virtual environment:

```powershell
# Windows (PowerShell)
git clone <repository-url>
cd RAG
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
git clone <repository-url>
cd RAG
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

Copy `.env.example` to `.env` and set your Google Gemini API key:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## 🔄 Running the Ingestion & Embedding Pipeline

*(Note: Pre-extracted chunks in `data/` and the `chroma_db/` index are included, but you can regenerate them anytime)*

Run the pipeline steps in order:

1. **Extract text from the source PDF**:
   ```powershell
   python ingestion/extract_text.py
   ```

2. **Chunk into structured articles**:
   ```powershell
   python ingestion/chunk_by_article.py
   ```

3. **Verify chunk quality & integrity**:
   ```powershell
   python scripts/check_chunks.py
   ```

4. **Build the persistent vector index in ChromaDB**:
   ```powershell
   python embeddings/build_index.py
   ```
   *(Loads both `data/articles_chunked.json` and `data/overview_chunks.json`, indexing 414 total vectors).*

---

## ⚡ Starting the FastAPI Backend

Launch the FastAPI backend with Uvicorn:

```powershell
uvicorn app.main:app --reload --port 8000
```

- **Base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`

### Quick API Verification

In another terminal, test the `/ask` endpoint using the test script:

```powershell
python scripts/test_ask.py
```

Or via `curl` / PowerShell:
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/ask" `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"question": "What does Article 21 say?"}'
```

**Response format**:
```json
{
  "answer": "According to Article 21, no person shall be deprived of his life or personal liberty except according to procedure established by law.",
  "cited_articles": ["21"],
  "retrieved_sources": [
    {
      "article_number": "21",
      "title": "Protection of life and personal liberty",
      "distance": 0.4662
    }
  ]
}
```

---

## 🖥️ Running the Streamlit Demo UI

With the FastAPI server running, launch the interactive Streamlit interface:

```powershell
streamlit run demo/streamlit_app.py
```

Navigate to `http://localhost:8501`. Features include:
- Backend health status indicator
- Clickable sample queries for common constitutional topics
- Formatted answers with verified article citation tags
- Collapsible source cards displaying retrieved context articles and cosine distances

---

## 📊 Evaluation Benchmark

Run the automated retrieval evaluation over the 25 benchmark QA pairs:

```powershell
python eval/run_eval.py
```

### Benchmark Results
- **Top-1 Retrieval Accuracy:** **76.0%** (19/25 questions match exact target article at Rank #1)
- **Top-5 Retrieval Accuracy:** **92.0%** (23/25 questions retrieve the target article within top 5)

---

## 🛡️ Guardrails & Known Limitations

1. **Strict Domain-Locking**: System prompt strictly forbids the model from extrapolating or using outside knowledge. All answers must be directly grounded in the retrieved excerpts.
2. **Deterministic Out-of-Scope Fallback**: If the cosine distance of the closest retrieved chunk exceeds `0.75`, the query is flagged as out-of-scope and the LLM call is bypassed entirely, returning `"I don't have information on that in the Constitution."`
3. **Broad/Vague Query Handling**: The bot is intentionally designed to decline broad, vague, or subjective questions that cannot be grounded in a specific constitutional article or overview chunk, rather than risk hallucinations.
4. **Overview Chunks**: To support summary questions that span multiple provisions (such as "What are the important articles for Indian citizens?"), curated overview chunks (`data/overview_chunks.json`) cover Fundamental Rights, Duties, Directive Principles, Government Structure, and Constitutional Amendments.
5. **Amendment Disambiguation**: Chunks originating from Amendment Acts are labeled with `source: amendment_act` and de-prioritized in favor of main body articles for identical article numbers.
