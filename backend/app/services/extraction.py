import re
from pathlib import Path
import pymupdf as fitz
from app.models.legal import PageText


_HEADER_FOOTER_WHITESPACE = re.compile(r"[ \t]+")


def _clean_page_text(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = _HEADER_FOOTER_WHITESPACE.sub(" ", raw).strip()
        if not line:
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    doc = fitz.open(pdf_path)
    pages: list[PageText] = []
    for index, page in enumerate(doc):
        text = _clean_page_text(page.get_text("text") or "")
        pages.append(PageText(page_number=index + 1, text=text))
    doc.close()
    return pages


def needs_ocr(pages: list[PageText], min_chars_per_page: int = 80) -> bool:
    if not pages:
        return True
    nonempty = sum(1 for page in pages if len(page.text.strip()) >= min_chars_per_page)
    return nonempty / len(pages) < 0.5
