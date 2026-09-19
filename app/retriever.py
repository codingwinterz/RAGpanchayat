"""Retriever module: hybrid search (vector + BM25) against the ChromaDB constitution index.

Combines ChromaDB cosine-similarity search with BM25 keyword search using
Reciprocal Rank Fusion (RRF).  Prioritises main_body articles over
amendment_act articles when both appear in the results for the same article
number.

The cosine-distance out-of-scope threshold (``SIMILARITY_THRESHOLD``) applies
to *vector* distances only.  Chunks surfaced exclusively by BM25
(``keyword_only``) are exempt: BM25 can recall exact legal terms that
embedding similarity misses, and discarding them solely because they lack a
vector distance would silently remove exactly the results keyword search is
useful for.
"""

import os
from typing import Any
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHROMA_DIR: str = "chroma_db"
COLLECTION_NAME: str = "constitution_articles"
MODEL_NAME: str = "all-MiniLM-L6-v2"

# Cosine distance threshold — lower is better (0 = identical).
# Relevant constitutional queries typically score 0.25 - 0.65.
# Out-of-scope queries (e.g. general knowledge, recipes) score > 0.75.
SIMILARITY_THRESHOLD: float = 0.75

# How many raw candidates to fetch from *each* retrieval method
# before RRF fusion. Wider pool gives RRF more signal.
_RAW_TOP_K: int = 20

# RRF constant (standard value from the original Cormack et al. paper).
_RRF_K: int = 60

# Toggle hybrid BM25 search (set ENABLE_BM25=0/false in the environment to
# fall back to pure vector retrieval).
ENABLE_BM25: bool = os.getenv("ENABLE_BM25", "1").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Module-level singletons (loaded once on first import)
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None
_collection: Any = None

# BM25 index + backing corpus
_bm25: BM25Okapi | None = None
_bm25_docs: list[str] | None = None
_bm25_metas: list[dict[str, Any]] | None = None


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


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer — works well for formal legal English."""
    return text.lower().split()


def _load_bm25() -> tuple[BM25Okapi, list[str], list[dict[str, Any]]]:
    """Lazy-load BM25 index over the full ChromaDB corpus.

    Fetches every document from the collection, tokenizes each one, and
    builds a BM25Okapi index.  Cached as module-level singletons.

    Returns:
        A tuple of (BM25Okapi index, list of document texts, list of metadata dicts).
    """
    global _bm25, _bm25_docs, _bm25_metas
    if _bm25 is None:
        _, collection = _load()
        all_data = collection.get(include=["documents", "metadatas"])
        _bm25_docs = all_data["documents"]
        _bm25_metas = all_data["metadatas"]
        tokenized_corpus = [_tokenize(doc) for doc in _bm25_docs]
        _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25, _bm25_docs, _bm25_metas


# ---------------------------------------------------------------------------
# Distance guardrail
# ---------------------------------------------------------------------------

def _passes_distance_filter(
    chunk: dict[str, Any], threshold: float = SIMILARITY_THRESHOLD
) -> bool:
    """Decide whether a candidate chunk may be returned to the caller.

    Keyword-only chunks (BM25 hits with no vector distance) always pass:
    their placeholder distance is meaningless, and BM25 recall is the whole
    point of the hybrid retriever.  Everything else must be within the
    cosine-distance out-of-scope threshold.

    Args:
        chunk: Candidate chunk dict with ``distance`` and ``keyword_only`` keys.
        threshold: Maximum allowed cosine distance (lower is better).

    Returns:
        True if the chunk may be surfaced to the caller.
    """
    if chunk.get("keyword_only"):
        return True
    return chunk["distance"] <= threshold


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion helper
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    vector_candidates: list[dict[str, Any]],
    bm25_candidates: list[dict[str, Any]],
    k: int = _RRF_K,
) -> list[dict[str, Any]]:
    """Merge two ranked result lists using Reciprocal Rank Fusion.

    For every unique chunk (keyed by ``(article_number, source)``), the RRF
    score is::

        rrf = 1 / (k + rank_vector) + 1 / (k + rank_bm25)

    Chunks present in only one list receive a penalty rank of
    ``max(len(vector), len(bm25)) + 1``.

    Args:
        vector_candidates: Ranked results from ChromaDB vector search.
        bm25_candidates: Ranked results from BM25 keyword search.
        k: RRF smoothing constant (default 60).

    Returns:
        Merged list of candidate dicts sorted by descending RRF score.
        Each dict carries an additional ``rrf_score`` key.
    """
    default_rank = max(len(vector_candidates), len(bm25_candidates)) + 1

    # Build rank maps keyed by (article_number, source, text_hash)
    def _key(c: dict[str, Any]) -> tuple[str, str, int]:
        return (c["article_number"], c["source"], hash(c["text"]))

    vec_rank: dict[tuple, int] = {}
    vec_by_key: dict[tuple, dict[str, Any]] = {}
    for rank, c in enumerate(vector_candidates, start=1):
        key = _key(c)
        if key not in vec_rank:          # keep best (first) rank
            vec_rank[key] = rank
            vec_by_key[key] = c

    bm25_rank: dict[tuple, int] = {}
    bm25_by_key: dict[tuple, dict[str, Any]] = {}
    for rank, c in enumerate(bm25_candidates, start=1):
        key = _key(c)
        if key not in bm25_rank:
            bm25_rank[key] = rank
            bm25_by_key[key] = c

    all_keys = set(vec_rank) | set(bm25_rank)
    fused: list[dict[str, Any]] = []
    for key in all_keys:
        r_vec = vec_rank.get(key, default_rank)
        r_bm25 = bm25_rank.get(key, default_rank)
        rrf_score = 1.0 / (k + r_vec) + 1.0 / (k + r_bm25)

        # Use the vector candidate if available (it carries the real distance),
        # otherwise fall back to the BM25-only candidate.
        candidate = dict(vec_by_key.get(key) or bm25_by_key[key])
        candidate["rrf_score"] = round(rrf_score, 6)
        candidate["keyword_only"] = key not in vec_rank
        fused.append(candidate)

    fused.sort(key=lambda c: (c["rrf_score"], -c["distance"]), reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed the question and return the top-k most relevant article chunks.

    Uses hybrid retrieval: ChromaDB vector similarity *and* BM25 keyword
    search are run independently, then merged with Reciprocal Rank Fusion.
    Main-body articles are boosted above amendment-act duplicates.

    Args:
        question: User query string to search for.
        top_k: Maximum number of filtered chunks to return. Defaults to 5.

    Returns:
        A list of dicts::

            [{"article_number": "21", "title": "...", "text": "...",
              "source": "main_body", "distance": 0.42,
              "rrf_score": 0.032, "keyword_only": False}, ...]
    """
    model, collection = _load()

    # ------------------------------------------------------------------
    # 1. Vector search (ChromaDB cosine similarity)
    # ------------------------------------------------------------------
    query_embedding = model.encode(question, convert_to_numpy=True).tolist()

    n_results = min(_RAW_TOP_K, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs_vec = results["documents"][0]
    metas_vec = results["metadatas"][0]
    dists_vec = results["distances"][0]

    vector_candidates: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs_vec, metas_vec, dists_vec):
        vector_candidates.append({
            "article_number": meta["article_number"],
            "title": meta["title"],
            "source": meta.get("source", "main_body"),
            "text": doc,
            "distance": round(dist, 4),
            "keyword_only": False,
        })

    if ENABLE_BM25:
        bm25, bm25_docs, bm25_metas = _load_bm25()

        # ------------------------------------------------------------------
        # 2. BM25 keyword search
        # ------------------------------------------------------------------
        query_tokens = _tokenize(question)
        bm25_scores = bm25.get_scores(query_tokens)

        # Pick top-_RAW_TOP_K indices by BM25 score
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:_RAW_TOP_K]

        bm25_candidates: list[dict[str, Any]] = []
        for idx in top_bm25_indices:
            idx = int(idx)
            if bm25_scores[idx] <= 0:
                break  # no point including zero-score docs
            meta = bm25_metas[idx]
            bm25_candidates.append({
                "article_number": meta["article_number"],
                "title": meta["title"],
                "source": meta.get("source", "main_body"),
                "text": bm25_docs[idx],
                "distance": 1.0,  # placeholder — vector distance is unknown for keyword-only hits
                "keyword_only": True,
            })

        # ------------------------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # ------------------------------------------------------------------
        candidates = _reciprocal_rank_fusion(vector_candidates, bm25_candidates)
    else:
        candidates = vector_candidates

    # ------------------------------------------------------------------
    # 4. Prioritise main_body over amendment_act
    # ------------------------------------------------------------------
    seen_main: set[str] = set()
    prioritised: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []

    for c in candidates:
        if c["source"] == "main_body":
            seen_main.add(c["article_number"])
            prioritised.append(c)
        else:
            if c["article_number"] in seen_main:
                deferred.append(c)
            else:
                prioritised.append(c)

    ranked = prioritised + deferred

    # ------------------------------------------------------------------
    # 5. Filter by threshold and trim to top_k
    # ------------------------------------------------------------------
    filtered = [c for c in ranked if _passes_distance_filter(c)]
    return filtered[:top_k]
