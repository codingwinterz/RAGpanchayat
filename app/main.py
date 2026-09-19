"""FastAPI application — Constitution of India RAG chatbot."""

import os
import sys
from typing import Any

# Add project root to path so `app.retriever` / `app.prompt` resolve correctly
# when running from the project root with `uvicorn app.main:app`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.retriever import retrieve
from app.prompt import generate_answer, NO_CONTEXT_ANSWER

# Restrict cross-origin access to the Streamlit demo (plus any extra origins
# provided via the ALLOWED_ORIGINS environment variable).  A wildcard origin
# combined with allow_credentials is unsafe and inconsistently handled by
# browsers, so the default is an explicit allow-list.
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Constitution of India RAG Chatbot",
    description=(
        "Ask questions about the Constitution of India and get answers with "
        "verified article citations."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """Request schema for querying the Constitution of India chatbot."""
    question: str


class RetrievedSource(BaseModel):
    """Metadata for one retrieved context chunk returned to the client."""
    article_number: str
    title: str
    distance: float | None = None
    retrieval: str = "hybrid"  # "hybrid" or "keyword_only"


class AskResponse(BaseModel):
    """Response schema: generated answer plus verified citation metadata."""
    answer: str
    cited_articles: list[str]
    verified_articles: list[str]
    unverified_articles: list[str]
    retrieved_sources: list[RetrievedSource] | None = None


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Answer a question about the Constitution of India.

    1. Run hybrid retrieval (vector + BM25 via RRF) for the question.
    2. If nothing relevant survives the retriever's distance guardrail,
       return a fixed fallback without calling the LLM.
    3. Otherwise pass the retrieved context to Gemini, then verify that every
       parsed article citation actually appears in the retrieved context.

    Args:
        req: An AskRequest instance with the user's question.

    Returns:
        An AskResponse with the answer, citation lists, and source metadata.
    """
    chunks = retrieve(req.question, top_k=5)

    # retrieve() already drops chunks beyond SIMILARITY_THRESHOLD, so an empty
    # result means even the closest raw vector match was out of scope.
    # Return the deterministic fallback without spending an LLM call.
    if not chunks:
        return AskResponse(
            answer=NO_CONTEXT_ANSWER,
            cited_articles=[],
            verified_articles=[],
            unverified_articles=[],
            retrieved_sources=[],
        )

    sources: list[RetrievedSource] = []
    for c in chunks:
        label = c["article_number"]
        if c.get("source") == "amendment_act":
            label += " (Amendment Act)"
        keyword_only = bool(c.get("keyword_only"))
        sources.append(RetrievedSource(
            article_number=label,
            title=c["title"],
            distance=None if keyword_only else c["distance"],
            retrieval="keyword_only" if keyword_only else "hybrid",
        ))

    result = generate_answer(req.question, chunks)

    return AskResponse(
        answer=result["answer"],
        cited_articles=result["cited_articles"],
        verified_articles=result["verified_articles"],
        unverified_articles=result["unverified_articles"],
        retrieved_sources=sources,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint to verify backend service availability.

    Returns:
        A dictionary with the operational status of the service.
    """
    return {"status": "ok"}
