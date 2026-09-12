"""FastAPI application — Constitution of India RAG chatbot."""

import sys
import os

# Add project root to path so `app.retriever` / `app.prompt` resolve correctly
# when running from the project root with `uvicorn app.main:app`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.retriever import retrieve, SIMILARITY_THRESHOLD
from app.prompt import generate_answer, NO_CONTEXT_ANSWER

app = FastAPI(
    title="Constitution of India RAG Chatbot",
    description="Ask questions about the Constitution of India and get answers with cited article numbers.",
    version="1.0.0",
)

# Allow CORS for the Streamlit frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    cited_articles: list[str]
    retrieved_sources: list[dict] | None = None


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Answer a question about the Constitution of India.

    1. Embed the question and retrieve top-k relevant article chunks.
    2. If the best match distance exceeds the threshold, return a fixed
       "I don't know" response without calling the LLM.
    3. Otherwise, pass the retrieved context + question to Gemini and
       return the generated answer with cited article numbers.
    """
    chunks = retrieve(req.question, top_k=5)

    # If no chunks returned or best match is too far, skip LLM
    if not chunks or chunks[0]["distance"] > SIMILARITY_THRESHOLD:
        return AskResponse(
            answer=NO_CONTEXT_ANSWER,
            cited_articles=[],
            retrieved_sources=[],
        )

    # Build source summaries for the response (without full text)
    sources = []
    for c in chunks:
        label = c["article_number"]
        if c.get("source") == "amendment_act":
            label += " (Amendment Act)"
        sources.append({
            "article_number": label,
            "title": c["title"],
            "distance": c["distance"],
        })

    result = generate_answer(req.question, chunks)

    return AskResponse(
        answer=result["answer"],
        cited_articles=result["cited_articles"],
        retrieved_sources=sources,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
