"""Chunk extracted Constitution text into structured article blocks."""

import re
import json
from typing import Pattern

INPUT_PATH: str = "data/constitution_raw.txt"
OUTPUT_PATH: str = "data/articles_chunked.json"

# Matches article headings like:
#   1. Name and territory of the Union.—(1) India, that is Bharat...
#   2A. Sikkim to be associated with the Union.—...
#
# The separator between title and body is an em dash (U+2014), but we also
# accept en dash (U+2013), minus sign (U+2212), and hyphen-minus (U+002D).
# All dash-like characters are referenced by Unicode escape codes to avoid
# encoding corruption from copy-pasting special characters.
#
# Key fix: Do NOT use re.DOTALL. With DOTALL, the greedy/lazy .* in the
# body_start group swallows the entire remainder of the file, producing
# only 1 match for the whole document. Instead, we match each article's
# first line and then extract the full body as the text between consecutive
# article start positions.
ARTICLE_PATTERN: Pattern[str] = re.compile(
    r'^(?:[0-9*]*\[)?'                      # optional footnote prefix e.g. 2[ or 1[
    r'(?P<num>\d{1,3}[A-Z]{0,2})\.'          # article number (e.g. 21, 21A, 243ZG)
    r'\s+'                                  # whitespace after number
    r'(?P<title>'
    r'(?:[^\n\u2013\u2014\u2212]|'          # chars on same line (excluding em/en-dashes)
    r'\n(?!\s*(?:(?:[0-9*]*\[)?\d{1,3}[A-Z]{0,2}\.|PART|CHAPTER|SCHEDULE|APPENDIX))' # newline if not next article/major heading
    r')+?'
    r')'
    r'\.'                                   # title ends with a period
    r'[\u2013\u2014\u2212\u002D]'          # dash separator (en/em/minus/hyphen)
    r'\s*(?P<body_start>.*)$',             # rest of that line
    re.MULTILINE
)

# The APPENDIX I section in the PDF marks the boundary between the main
# Constitution body (Preamble + Articles 1-395 + Schedules) and the
# standalone amendment act texts appended at the end.
APPENDIX_MARKER: str = "APPENDIX I"

FOOTNOTE_MARKERS: list[str] = ["Subs.", "Ins.", "Rep.", "w.e.f.", "ibid", "Omitted"]


def is_footnote(title: str, body_preview: str) -> bool:
    """Return True if the match looks like a footnote rather than a real article.

    Args:
        title: Extracted heading title string.
        body_preview: First portion of the body text.

    Returns:
        True if suspect footnote markers are found in the snippet.
    """
    combined: str = (title + " " + body_preview[:80]).lower()
    return any(marker.lower() in combined for marker in FOOTNOTE_MARKERS)


def find_appendix_boundary(text: str) -> int:
    """Find character offset where APPENDIX I starts (amendment acts section).

    We look for the *last* occurrence of 'APPENDIX I' at the start of a line,
    which is the actual appendix content (not a TOC reference).

    Args:
        text: Full raw text of the Constitution.

    Returns:
        Character index of the start of Appendix I, or length of text if not found.
    """
    matches = list(re.finditer(r'^APPENDIX\s+I\b', text, re.MULTILINE))
    if matches:
        return matches[-1].start()
    return len(text)


def chunk_articles(text: str) -> list[dict[str, str]]:
    """Parse raw Constitution text into article chunks with metadata.

    Args:
        text: Extracted raw text of the Constitution.

    Returns:
        A list of chunk dictionaries with article_number, title, source, and text.
    """
    appendix_pos: int = find_appendix_boundary(text)
    matches = list(ARTICLE_PATTERN.finditer(text))
    chunks: list[dict[str, str]] = []

    for i, match in enumerate(matches):
        num: str = match.group("num")
        raw_title: str = match.group("title").strip()

        # Clean footnote bracket markers from title if present (e.g. "1[Power of..." -> "Power of...")
        clean_title: str = re.sub(r'^[0-9*]*\[', '', raw_title)
        if clean_title.endswith(']'):
            clean_title = clean_title[:-1].strip()
        clean_title = ' '.join(clean_title.split())

        # Body = from this match's start to the next match's start
        start: int = match.start()
        end: int = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        full_text: str = text[start:end].strip()

        if is_footnote(clean_title, full_text):
            continue

        # Determine source based on position relative to appendix boundary
        source: str = "amendment_act" if start >= appendix_pos else "main_body"

        chunks.append({
            "article_number": num,
            "title": clean_title,
            "source": source,
            "text": full_text
        })

    return chunks


def main() -> None:
    """Extract article chunks from constitution_raw.txt and save to articles_chunked.json."""
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_text: str = f.read()

    chunks: list[dict[str, str]] = chunk_articles(raw_text)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    main_count: int = sum(1 for c in chunks if c["source"] == "main_body")
    amend_count: int = sum(1 for c in chunks if c["source"] == "amendment_act")

    print(f"Extracted {len(chunks)} article chunks.")
    print(f"  Main body:      {main_count}")
    print(f"  Amendment acts: {amend_count}")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nFirst 3 chunks:")
    for c in chunks[:3]:
        print(f"\n[{c['source']}] Article {c['article_number']}: {c['title']}")
        print(c["text"][:200] + "...")
    print("\nLast 3 chunks:")
    for c in chunks[-3:]:
        print(f"\n[{c['source']}] Article {c['article_number']}: {c['title']}")
        print(c["text"][:200] + "...")


if __name__ == "__main__":
    main()