"""Evaluation runner for Constitution of India RAG.

Evaluates retrieval precision against hand-crafted QA benchmark pairs
in eval/qa_pairs.json.
"""

import sys
import os
import json
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever import retrieve

# Ensure UTF-8 stdout if possible on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

QA_PAIRS_PATH: str = os.path.join(os.path.dirname(__file__), "qa_pairs.json")


def evaluate_retrieval(top_k: int = 5) -> dict[str, Any] | None:
    """Run retrieval evaluation against qa_pairs.json.

    Args:
        top_k: Number of retrieved chunks to consider for recall calculation. Defaults to 5.

    Returns:
        A dictionary containing overall metrics (total, hits, accuracy percentages,
        and per-question result details), or None if the benchmark file is not found.
    """
    if not os.path.exists(QA_PAIRS_PATH):
        print(f"Error: QA benchmark file not found at {QA_PAIRS_PATH}")
        return None

    with open(QA_PAIRS_PATH, "r", encoding="utf-8") as f:
        qa_pairs: list[dict[str, Any]] = json.load(f)

    print("=" * 80)
    print(f"CONSTITUTION OF INDIA RAG - RETRIEVAL EVALUATION (top-{top_k})")
    print(f"Benchmark: {len(qa_pairs)} questions from {QA_PAIRS_PATH}")
    print("=" * 80)

    hits_top1: int = 0
    hits_topk: int = 0
    results: list[dict[str, Any]] = []

    for item in qa_pairs:
        qid: int = item["id"]
        q: str = item["question"]
        expected: list[str] = item["expected_articles"]

        retrieved: list[dict[str, Any]] = retrieve(q, top_k=top_k)
        retrieved_articles: list[str] = [r["article_number"] for r in retrieved]

        is_top1: bool = bool(retrieved_articles and retrieved_articles[0] in expected)
        is_topk: bool = any(art in expected for art in retrieved_articles)

        if is_top1:
            hits_top1 += 1
        if is_topk:
            hits_topk += 1

        status: str = "[PASS]" if is_topk else "[FAIL]"
        rank: str = (
            f"Rank #{retrieved_articles.index(next(a for a in retrieved_articles if a in expected)) + 1}"
            if is_topk
            else "Not in top-" + str(top_k)
        )

        results.append({
            "id": qid,
            "question": q,
            "expected": expected,
            "retrieved": retrieved_articles,
            "top1_hit": is_top1,
            "topk_hit": is_topk,
            "status": status,
            "rank": rank,
            "top_match": f"Art {retrieved[0]['article_number']}: {retrieved[0]['title'][:30]} (dist: {retrieved[0]['distance']})" if retrieved else "None",
        })

        print(f"[{status}] Q{qid:02d}: {q[:55]}...")
        print(f"       Expected: {expected} | Retrieved: {retrieved_articles} ({rank})")

    total: int = len(qa_pairs)
    top1_pct: float = (hits_top1 / total) * 100
    topk_pct: float = (hits_topk / total) * 100

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Benchmark Questions:   {total}")
    print(f"Top-1 Retrieval Accuracy:    {hits_top1}/{total} ({top1_pct:.1f}%)")
    print(f"Top-{top_k} Retrieval Accuracy:    {hits_topk}/{total} ({topk_pct:.1f}%)")
    print("=" * 80)

    return {
        "total": total,
        "hits_top1": hits_top1,
        "hits_topk": hits_topk,
        "top1_pct": top1_pct,
        "topk_pct": topk_pct,
        "details": results,
    }


if __name__ == "__main__":
    evaluate_retrieval(top_k=5)
