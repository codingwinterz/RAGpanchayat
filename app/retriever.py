"""Retriever module: similarity search against the ChromaDB constitution index.

Prioritises main_body articles over amendment_act articles when both appear
in the results for the same article number.
"""

from typing import Any
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR: str = "chroma_db"
COLLECTION_NAME: str = "constitution_articles"
MODEL_NAME: str = "all-MiniLM-L6-v2"

# Cosine distance threshold — lower is better (0 = identical).
# Relevant constitutional queries typically score 0.25 - 0.65.
# Out-of-scope queries (e.g. general knowledge, recipes) score > 0.75.
SIMILARITY_THRESHOLD: float = 0.75

# How many raw candidates to fetch before filtering/reranking.
_RAW_TOP_K: int = 10


# Module-level singletons (loaded once on first import)
_model: SentenceTransformer | None = None
_collection: Any = None


def _load() -> tuple[SentenceTransformer, Any]:
    """Lazy-load the embedding model and ChromaDB collection singleton instances.

    Returns:
        A tuple of (SentenceTransformer model, ChromaDB collection).
    """
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _model, _collection


def retrieve(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed the question and return the top-k most relevant article chunks.

    Main-body articles are boosted above amendment-act duplicates.

    Args:
        question: User query string to search for.
        top_k: Maximum number of filtered chunks to return. Defaults to 5.

    Returns:
        A list of dicts:
            [{"article_number": "21", "title": "...", "text": "...",
              "source": "main_body", "distance": 0.42}, ...]
    """
    model, collection = _load()

    query_embedding = model.encode(question, convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(_RAW_TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested list structure
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    candidates: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        candidates.append({
            "article_number": meta["article_number"],
            "title": meta["title"],
            "source": meta.get("source", "main_body"),
            "text": doc,
            "distance": round(dist, 4),
        })

    # --- Prioritise main_body over amendment_act ---
    # For each article number, if both sources appear, keep main_body and
    # push amendment_act to the end (rather than removing it entirely).
    seen_main: set[str] = set()
    prioritised: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []

    for c in candidates:
        if c["source"] == "main_body":
            seen_main.add(c["article_number"])
            prioritised.append(c)
        else:
            # If we already have a main_body hit for this article number,
            # defer the amendment-act duplicate.
            if c["article_number"] in seen_main:
                deferred.append(c)
            else:
                prioritised.append(c)

    ranked = prioritised + deferred

    # Filter by threshold and trim to top_k
    filtered = [c for c in ranked if c["distance"] <= SIMILARITY_THRESHOLD]
    return filtered[:top_k]
