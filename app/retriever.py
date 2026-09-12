"""Retriever module: similarity search against the ChromaDB constitution index.

Prioritises main_body articles over amendment_act articles when both appear
in the results for the same article number.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "constitution_articles"
MODEL_NAME = "all-MiniLM-L6-v2"

# Cosine distance threshold — lower is better (0 = identical).
# Relevant constitutional queries typically score 0.25 - 0.65.
# Out-of-scope queries (e.g. general knowledge, recipes) score > 0.75.
SIMILARITY_THRESHOLD = 0.75

# How many raw candidates to fetch before filtering/reranking.
_RAW_TOP_K = 10


# Module-level singletons (loaded once on first import)
_model = None
_collection = None


def _load():
    """Lazy-load the embedding model and ChromaDB collection."""
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _model, _collection


def retrieve(question: str, top_k: int = 5):
    """Embed the question and return the top-k most relevant article chunks.

    Returns a list of dicts:
        [{"article_number": "21", "title": "...", "text": "...",
          "source": "main_body", "distance": 0.42}, ...]

    Main-body articles are boosted above amendment-act duplicates.
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

    candidates = []
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
    seen_main = set()
    prioritised = []
    deferred = []

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
