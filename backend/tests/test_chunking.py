from app.models.legal import LegalMetadata, PageText
from app.services.chunking import chunk_document


def test_statute_section_traceability():
    pages = [
        PageText(page_number=1, text="CHAPTER I\n1. Short title and commencement\nThis Act may be called the Example Act."),
        PageText(page_number=2, text="2. Definitions\nIn this Act, unless the context otherwise requires..."),
    ]
    meta = LegalMetadata(title="Example Act", document_type="statute", jurisdiction="India")
    chunks = chunk_document("DOC_TEST", pages, meta)
    sections = {c.section for c in chunks}
    assert "1" in sections
    assert "2" in sections
    assert all(c.page_start <= c.page_end for c in chunks)
