"""Prompt construction and Gemini generation for Constitution Q&A."""

import os
import re
from typing import Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root

# Model identifier used for generation. Defaults to the project's pinned
# model; override via the GEMINI_MODEL environment variable if needed.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

_client_configured: bool = False
_gemini_model: genai.GenerativeModel | None = None

SYSTEM_PROMPT = (
    "You are a legal expert on the Constitution of India. "
    "Answer the user's question using ONLY the provided context excerpts. "
    "Cite the specific Article number(s) you relied on (e.g. 'Article 21'). "
    "Only cite articles that appear in the provided context excerpts. "
    "If the answer is not in the provided context, reply exactly: "
    "'I don't have information on that in the Constitution.'\n\n"
    "IMPORTANT: If any context excerpt is from an amendment act "
    "(indicated by [AMENDMENT ACT] label), clearly state that in your answer "
    "so the user knows which source the information comes from."
)

NO_CONTEXT_ANSWER: str = "I don't have information on that in the Constitution."

# One citation item: an article number with optional compound letter suffix
# ("21", "243ZG") followed by optional sub-article clauses ("19(1)(a)").
_CITATION_ITEM = r'\d{1,3}[A-Z]{0,2}(?:\([^)]*\))*'

# Full citation: keyword ("Article"/"Art."/"Arts."), first item, then any
# number of separator+item groups (", 34", " and 21", " or 35") for lists.
_CITATION_PATTERN = re.compile(
    r'[Aa]rt(?:icle)?s?\b\s*\.?\s*'
    + _CITATION_ITEM
    + r'(?:(?:,\s*|\s+and\s+|\s+or\s+)' + _CITATION_ITEM + r')*'
)

# Sub-article numerals inside parentheses (e.g. the "(1)(a)" in 19(1)(a))
_PAREN_GROUP = re.compile(r'\([^)]*\)')

# The article-number token itself
_NUMBER_TOKEN = re.compile(r'\d{1,3}[A-Z]{0,2}')


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
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
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

    Handles singular and plural keyword forms ("Article 21", "Art. 32",
    "Arts. 14"), comma-, "and"- and "or"-separated lists ("Articles 14 and
    21", "Articles 32, 34 and 35"), sub-article references ("Article
    19(1)(g)" yields "19"), and compound suffixes ("Article 243ZG").

    Args:
        answer_text: Text response returned by the language model.

    Returns:
        Deduplicated list of article identifier strings, in order of first
        appearance.
    """
    seen: set[str] = set()
    cited: list[str] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        # Drop sub-article parentheses so their numerals are not mistaken
        # for standalone article numbers ("19(1)(a)" -> "19", not "19, 1").
        citation_text = _PAREN_GROUP.sub(" ", match.group(0))
        for raw in _NUMBER_TOKEN.findall(citation_text):
            if raw not in seen:
                seen.add(raw)
                cited.append(raw)
    return cited


def _verify_citations(
    citations: list[str], retrieved_articles: set[str]
) -> tuple[list[str], list[str]]:
    """Split parsed citations into verified and unverified groups.

    A citation is *verified* when the cited article number exists in the
    retrieved context that was sent to the LLM.  Anything else may be a
    hallucinated citation and is reported separately so clients can flag it.

    Args:
        citations: Article identifiers parsed from the LLM's answer.
        retrieved_articles: Article numbers present in the retrieved context.

    Returns:
        A tuple of (verified_articles, unverified_articles).
    """
    verified = [c for c in citations if c in retrieved_articles]
    unverified = [c for c in citations if c not in retrieved_articles]
    return verified, unverified


def generate_answer(question: str, context_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate an answer using Gemini, grounded in the retrieved context.

    Args:
        question: The user query string.
        context_chunks: List of retrieved article dictionaries containing text and metadata.

    Returns:
        Dictionary containing:
            - "answer": String response from the LLM or fallback message.
            - "cited_articles": All article identifiers parsed from the answer.
            - "verified_articles": Citations present in the retrieved context.
            - "unverified_articles": Citations absent from the retrieved context
              (possible hallucinations, flagged for the client).
    """
    if not context_chunks:
        return {
            "answer": NO_CONTEXT_ANSWER,
            "cited_articles": [],
            "verified_articles": [],
            "unverified_articles": [],
        }

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
    retrieved_articles = {c["article_number"] for c in context_chunks}
    verified, unverified = _verify_citations(cited, retrieved_articles)

    return {
        "answer": answer_text,
        "cited_articles": cited,
        "verified_articles": verified,
        "unverified_articles": unverified,
    }
