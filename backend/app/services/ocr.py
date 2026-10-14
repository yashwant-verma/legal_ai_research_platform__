from __future__ import annotations
import io
from pathlib import Path
import pymupdf as fitz
from app.models.legal import PageText

def ocr_pdf_pages(pdf_path: Path, dpi: int = 220, lang: str = 'eng') -> list[PageText]:
    """OCR fallback. Requires the Tesseract binary on the host.
    For Hindi, install the `hin` language pack and use lang='eng+hin'.
    """
    from PIL import Image
    import pytesseract
    doc=fitz.open(pdf_path); out=[]
    scale=dpi/72.0; matrix=fitz.Matrix(scale,scale)
    try:
        for i,page in enumerate(doc):
            pix=page.get_pixmap(matrix=matrix,alpha=False)
            image=Image.open(io.BytesIO(pix.tobytes('png')))
            text=pytesseract.image_to_string(image,lang=lang,config='--psm 6').strip()
            out.append(PageText(page_number=i+1,text=text))
    finally:
        doc.close()
    return out
