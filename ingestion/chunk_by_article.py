import re
import json

INPUT_PATH = "data/constitution_raw.txt"
OUTPUT_PATH = "data/articles_chunked.json"

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
ARTICLE_PATTERN = re.compile(
    r'^(?P<num>\d{1,3}[A-Z]?)\.'           # article number at start of line
    r'\s+(?P<title>[^.]+?)\.'               # title (up to next period)
    r'[\u2013\u2014\u2212\u002D]'           # dash separator (en/em/minus/hyphen)
    r'\s*(?P<body_start>.*)$',              # rest of that first line
    re.MULTILINE
)

# The APPENDIX I section in the PDF marks the boundary between the main
# Constitution body (Preamble + Articles 1-395 + Schedules) and the
# standalone amendment act texts appended at the end.
APPENDIX_MARKER = "APPENDIX I"

FOOTNOTE_MARKERS = ["Subs.", "Ins.", "Rep.", "w.e.f.", "ibid", "Omitted"]


def is_footnote(title, body_preview):
    """Return True if the match looks like a footnote rather than a real article."""
    combined = (title + " " + body_preview[:80]).lower()
    return any(marker.lower() in combined for marker in FOOTNOTE_MARKERS)


def find_appendix_boundary(text):
    """Find where APPENDIX I starts (amendment acts section).

    We look for the *last* occurrence of 'APPENDIX I' at the start of a line,
    which is the actual appendix content (not a TOC reference).
    """
    # Find all line-start occurrences of APPENDIX I
    matches = list(re.finditer(r'^APPENDIX\s+I\b', text, re.MULTILINE))
    if matches:
        # Use the last substantial occurrence (the actual appendix, not TOC)
        return matches[-1].start()
    return len(text)  # fallback: treat everything as main body


def chunk_articles(text):
    appendix_pos = find_appendix_boundary(text)
    matches = list(ARTICLE_PATTERN.finditer(text))
    chunks = []

    for i, match in enumerate(matches):
        num = match.group("num")
        title = match.group("title").strip()

        # Body = from this match's start to the next match's start
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        full_text = text[start:end].strip()

        if is_footnote(title, full_text):
            continue

        # Determine source based on position relative to appendix boundary
        source = "amendment_act" if start >= appendix_pos else "main_body"

        chunks.append({
            "article_number": num,
            "title": title,
            "source": source,
            "text": full_text
        })

    return chunks


if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = chunk_articles(raw_text)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    main_count = sum(1 for c in chunks if c["source"] == "main_body")
    amend_count = sum(1 for c in chunks if c["source"] == "amendment_act")

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