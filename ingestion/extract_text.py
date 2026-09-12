"""Extract raw text from official Constitution of India PDF using pdfplumber."""

# pyrefly: ignore [missing-import]
import pdfplumber

PDF_PATH: str = "data/constitution.pdf"
OUTPUT_PATH: str = "data/constitution_raw.txt"


def extract_text() -> None:
    """Extract page-by-page text from constitution.pdf and write to constitution_raw.txt."""
    full_text: str = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            text: str | None = page.extract_text()
            if text:
                full_text += text + "\n"
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} pages...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Done. Extracted text saved to {OUTPUT_PATH}")
    print(f"Total characters: {len(full_text)}")


if __name__ == "__main__":
    extract_text()