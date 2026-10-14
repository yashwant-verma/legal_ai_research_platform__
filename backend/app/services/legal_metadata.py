import json
import re
from pathlib import Path
from app.models.legal import LegalMetadata, PageText


COURT_PATTERNS = [
    ("Supreme Court of India", re.compile(r"SUPREME COURT OF INDIA", re.I)),
    ("High Court", re.compile(r"HIGH COURT OF", re.I)),
]


def _joined_prefix(pages: list[PageText], chars: int = 12000) -> str:
    return "\n".join(p.text for p in pages)[:chars]


def infer_metadata(pdf_path: Path, pages: list[PageText]) -> LegalMetadata:
    text = _joined_prefix(pages)
    name = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
    lowered = text.lower()

    document_type = "other"
    if "constitution of india" in lowered:
        document_type = "constitution"
    elif re.search(r"\bact,?\s+(18|19|20)\d{2}\b", text, re.I):
        document_type = "statute"
    elif "judgment" in lowered or "petitioner" in lowered and "respondent" in lowered:
        document_type = "judgment"
    elif "notification" in lowered:
        document_type = "notification"
    elif re.search(r"\brules?,?\s+(18|19|20)\d{2}\b", text, re.I):
        document_type = "rule"

    court = None
    for court_name, pattern in COURT_PATTERNS:
        if pattern.search(text):
            court = court_name
            break

    year_match = re.search(r"\b((?:18|19|20)\d{2})\b", text)
    year = int(year_match.group(1)) if year_match else None

    notes = [
        "Metadata was inferred conservatively from filename/text.",
        "Verify jurisdiction, source, temporal validity and legal status before final benchmark use.",
    ]

    return LegalMetadata(
        title=name or None,
        document_type=document_type,
        jurisdiction="India" if "india" in lowered else None,
        court=court,
        year=year,
        language="en",
        metadata_source="heuristic",
        metadata_review_required=True,
        metadata_notes=notes,
    )


def load_sidecar_metadata(pdf_path: Path) -> LegalMetadata | None:
    sidecar = pdf_path.with_suffix(".metadata.json")
    if not sidecar.exists():
        return None
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata = LegalMetadata.model_validate(data)
    metadata.metadata_source = "sidecar"
    # Sidecar can explicitly set review_required. If omitted, require review for safety.
    return metadata


def resolve_metadata(pdf_path: Path, pages: list[PageText]) -> LegalMetadata:
    heuristic = infer_metadata(pdf_path, pages)
    sidecar = load_sidecar_metadata(pdf_path)
    if not sidecar:
        return heuristic

    merged = heuristic.model_dump()
    sidecar_data = sidecar.model_dump(exclude_none=True)
    merged.update(sidecar_data)
    merged["metadata_source"] = "mixed"
    return LegalMetadata.model_validate(merged)
