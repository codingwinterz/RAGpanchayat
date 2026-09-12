"""Diagnostic script to verify the quality and integrity of articles_chunked.json."""

import json
from collections import Counter
from typing import Any

INPUT_PATH: str = "data/articles_chunked.json"

FOOTNOTE_WORDS: list[str] = ["subs.", "ins.", "rep.", "w.e.f.", "ibid", "omitted"]


def main() -> None:
    """Run validation checks on chunk count, duplicates, short entries, and footnote leakage."""
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks: list[dict[str, Any]] = json.load(f)

    print(f"Total chunks: {len(chunks)}")
    main_count: int = sum(1 for c in chunks if c.get("source") == "main_body")
    amend_count: int = sum(1 for c in chunks if c.get("source") == "amendment_act")
    print(f"  Main body:      {main_count}")
    print(f"  Amendment acts: {amend_count}")

    # --- Article number distribution ---
    nums: list[str] = [c["article_number"] for c in chunks]
    num_counts: Counter[str] = Counter(nums)
    duplicates: dict[str, int] = {n: cnt for n, cnt in num_counts.items() if cnt > 1}
    print(f"\nUnique article numbers: {len(num_counts)}")
    if duplicates:
        print(f"Duplicated numbers ({len(duplicates)}):")
        for n, cnt in sorted(duplicates.items(),
                             key=lambda x: (int(x[0].rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")), x[0])):
            sources = [c["source"] for c in chunks if c["article_number"] == n]
            print(f"  Article {n}: {cnt}x  sources={sources}")

    # --- First and last 5 ---
    print("\n--- First 5 chunks ---")
    for c in chunks[:5]:
        print(f"  [{c.get('source','?')}] Art {c['article_number']}: "
              f"{c['title'][:60]}  ({len(c['text'])} chars)")

    print("\n--- Last 5 chunks ---")
    for c in chunks[-5:]:
        print(f"  [{c.get('source','?')}] Art {c['article_number']}: "
              f"{c['title'][:60]}  ({len(c['text'])} chars)")

    # --- Short chunks (possible TOC junk) ---
    short: list[dict[str, Any]] = [c for c in chunks if len(c["text"]) < 50]
    if short:
        print(f"\n--- Short chunks (< 50 chars): {len(short)} ---")
        for c in short:
            print(f"  Art {c['article_number']}: {c['text']!r}")
    else:
        print("\nNo short chunks (< 50 chars) found.")

    # --- Footnote-like chunks that slipped through ---
    suspect: list[dict[str, Any]] = []
    for c in chunks:
        preview: str = (c["title"] + " " + c["text"][:80]).lower()
        if any(fw in preview for fw in FOOTNOTE_WORDS):
            suspect.append(c)
    if suspect:
        print(f"\n--- Possible footnote chunks: {len(suspect)} ---")
        for c in suspect[:10]:
            print(f"  Art {c['article_number']}: {c['title'][:60]}")
    else:
        print("\nNo footnote-like chunks detected.")

    # --- Numeric coverage check for main body ---
    main_nums: list[int] = sorted(set(
        int(c["article_number"].rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        for c in chunks if c.get("source") == "main_body"
    ))
    if main_nums:
        expected = set(range(1, main_nums[-1] + 1))
        missing = expected - set(main_nums)
        # Some articles were repealed (e.g., 31, 69A) so gaps are expected
        print(f"\nMain body numeric range: {main_nums[0]} - {main_nums[-1]}")
        if missing:
            print(f"Missing base numbers ({len(missing)}): "
                  f"{sorted(missing)[:30]}{'...' if len(missing) > 30 else ''}")

    # --- Text length statistics ---
    lengths: list[int] = [len(c["text"]) for c in chunks]
    print(f"\nText length stats:")
    print(f"  Min:    {min(lengths)} chars")
    print(f"  Max:    {max(lengths)} chars")
    print(f"  Mean:   {sum(lengths) // len(lengths)} chars")
    print(f"  Median: {sorted(lengths)[len(lengths) // 2]} chars")

    print("\n--- VERDICT ---")
    if 350 <= len(chunks) <= 450 and not short:
        print("PASS: Chunk count and quality look good.")
    else:
        issues: list[str] = []
        if not (350 <= len(chunks) <= 450):
            issues.append(f"count {len(chunks)} outside 350-450 range")
        if short:
            issues.append(f"{len(short)} short chunks detected")
        print(f"REVIEW NEEDED: {', '.join(issues)}")


if __name__ == "__main__":
    main()
