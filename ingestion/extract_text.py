# pyrefly: ignore [missing-import]
import pdfplumber

PDF_PATH = "data/constitution.pdf"
OUTPUT_PATH = "data/constitution_raw.txt"

def extract_text():
    full_text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
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