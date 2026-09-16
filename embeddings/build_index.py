"""Build a persistent ChromaDB vector index from the chunked articles."""

import json
import time
import os
from typing import Any
import chromadb
from sentence_transformers import SentenceTransformer

ARTICLES_PATH: str = "data/articles_chunked.json"
OVERVIEW_PATH: str = "data/overview_chunks.json"
CHROMA_DIR: str = "chroma_db"
COLLECTION_NAME: str = "constitution_articles"
MODEL_NAME: str = "all-MiniLM-L6-v2"
BATCH_SIZE: int = 100


def main() -> None:
    """Load chunked articles and overview summaries, embed with sentence-transformers, and store in ChromaDB."""
    # --- Load chunks ---
    print("Loading chunked articles...")
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        chunks: list[dict[str, Any]] = json.load(f)
    print(f"  Loaded {len(chunks)} article chunks.")

    if os.path.exists(OVERVIEW_PATH):
        print("Loading overview chunks...")
        with open(OVERVIEW_PATH, "r", encoding="utf-8") as f:
            overview_chunks: list[dict[str, Any]] = json.load(f)
        chunks.extend(overview_chunks)
        print(f"  Loaded {len(overview_chunks)} overview chunks.")

    print(f"  Total chunks to index: {len(chunks)}.")

    # --- Load embedding model ---
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("  Model loaded.")

    # --- Initialise persistent ChromaDB ---
    print(f"Initialising ChromaDB at ./{CHROMA_DIR}/...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (to rebuild cleanly)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Created collection '{COLLECTION_NAME}'.")

    # --- Embed and insert in batches ---
    total: int = len(chunks)
    start_time: float = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end: int = min(batch_start + BATCH_SIZE, total)
        batch: list[dict[str, Any]] = chunks[batch_start:batch_end]

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, chunk in enumerate(batch):
            global_idx: int = batch_start + idx
            clean_art: str = str(chunk["article_number"]).replace(" ", "_")
            doc_id: str = f"doc_{clean_art}_{global_idx}"
            ids.append(doc_id)
            documents.append(chunk["text"])
            metadatas.append({
                "article_number": chunk["article_number"],
                "title": chunk["title"],
                "source": chunk.get("source", "main_body"),
            })

        # Embed the batch
        embeddings: list[list[float]] = model.encode(
            documents, show_progress_bar=False, convert_to_numpy=True
        ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        elapsed: float = time.time() - start_time
        print(f"  Indexed {batch_end}/{total} chunks "
              f"({batch_end / total * 100:.0f}%) [{elapsed:.1f}s]")

    elapsed = time.time() - start_time
    print(f"\nDone. Indexed {collection.count()} articles in {elapsed:.1f}s.")
    print(f"ChromaDB persisted to ./{CHROMA_DIR}/")


if __name__ == "__main__":
    main()
