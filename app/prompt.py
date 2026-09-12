"""Prompt construction and Gemini generation for Constitution Q&A."""

import os
import re
from typing import Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root

_client_configured: bool = False
_gemini_model: genai.GenerativeModel | None = None

SYSTEM_PROMPT = (
    "You are a legal expert on the Constitution of India. "
    "Answer the user's question using ONLY the provided context excerpts. "
    "Cite the specific Article number(s) you relied on (e.g. 'Article 21'). "
    "If the answer is not in the provided context, reply exactly: "
    "'I don't have information on that in the Constitution.'\n\n"
    "IMPORTANT: If any context excerpt is from an amendment act "
    "(indicated by [AMENDMENT ACT] label), clearly state that in your answer "
    "so the user knows which source the information comes from."
)

NO_CONTEXT_ANSWER: str = "I don't have information on that in the Constitution."


def _setup() -> genai.GenerativeModel:
    """Configure the Gemini client (once) and return the GenerativeModel instance.

    Returns:
        The configured genai.GenerativeModel singleton instance.

    Raises:
        RuntimeError: If GEMINI_API_KEY environment variable is not found.
    """
    global _client_configured, _gemini_model
    if not _client_configured:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Create a .env file or export it in your shell."
            )
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-3.6-flash")
        _client_configured = True
    return _gemini_model


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the LLM.

    Args:
        chunks: List of retrieved article chunks with metadata and text.

    Returns:
        Formatted context string with article headers and text blocks.
    """
    parts: list[str] = []
    for c in chunks:
        source_label = ""
        if c.get("source") == "amendment_act":
            source_label = " [AMENDMENT ACT]"
        header = (
            f"--- Article {c['article_number']}: "
            f"{c['title']}{source_label} ---"
        )
        parts.append(f"{header}\n{c['text']}")
    return "\n\n".join(parts)


def _extract_cited_articles(answer_text: str) -> list[str]:
    """Parse article numbers mentioned in the LLM's answer.

    Args:
        answer_text: Text response returned by the language model.

    Returns:
        Deduplicated list of article identifier strings cited in the text.
    """
    # Match patterns like "Article 21", "Article 19(1)", "Article 368",
    # "Articles 14 and 21", "Art. 32"
    pattern = r'[Aa]rticles?\s*\.?\s*(\d{1,3}[A-Z]?)'
    matches = re.findall(pattern, answer_text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    cited: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            cited.append(m)
    return cited


def generate_answer(question: str, context_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate an answer using Gemini, grounded in the retrieved context.

    Args:
        question: The user query string.
        context_chunks: List of retrieved article dictionaries containing text and metadata.

    Returns:
        Dictionary containing:
            - "answer": String response from the LLM or fallback message.
            - "cited_articles": List of article identifiers parsed from the answer.
    """
    if not context_chunks:
        return {"answer": NO_CONTEXT_ANSWER, "cited_articles": []}

    model = _setup()
    context = _build_context_block(context_chunks)

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = model.generate_content(
        contents=[
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
            {"role": "model", "parts": [{"text": "Understood. I will answer only from the provided context and cite article numbers."}]},
            {"role": "user", "parts": [{"text": user_message}]},
        ],
    )

    answer_text = response.text.strip()
    cited = _extract_cited_articles(answer_text)

    return {"answer": answer_text, "cited_articles": cited}
