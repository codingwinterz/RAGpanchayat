"""Build a persistent ChromaDB vector index from the chunked articles."""

import json
import time
import chromadb
from sentence_transformers import SentenceTransformer

import os

ARTICLES_PATH = "data/articles_chunked.json"
OVERVIEW_PATH = "data/overview_chunks.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "constitution_articles"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 100


def main():
    # --- Load chunks ---
    print("Loading chunked articles...")
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} article chunks.")

    if os.path.exists(OVERVIEW_PATH):
        print("Loading overview chunks...")
        with open(OVERVIEW_PATH, "r", encoding="utf-8") as f:
            overview_chunks = json.load(f)
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
    total = len(chunks)
    start_time = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = chunks[batch_start:batch_end]

        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(batch):
            global_idx = batch_start + idx
            clean_art = str(chunk["article_number"]).replace(" ", "_")
            doc_id = f"doc_{clean_art}_{global_idx}"
            ids.append(doc_id)
            documents.append(chunk["text"])
            metadatas.append({
                "article_number": chunk["article_number"],
                "title": chunk["title"],
                "source": chunk.get("source", "main_body"),
            })

        # Embed the batch
        embeddings = model.encode(
            documents, show_progress_bar=False, convert_to_numpy=True
        ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        elapsed = time.time() - start_time
        print(f"  Indexed {batch_end}/{total} chunks "
              f"({batch_end / total * 100:.0f}%) [{elapsed:.1f}s]")

    elapsed = time.time() - start_time
    print(f"\nDone. Indexed {collection.count()} articles in {elapsed:.1f}s.")
    print(f"ChromaDB persisted to ./{CHROMA_DIR}/")


if __name__ == "__main__":
    main()
