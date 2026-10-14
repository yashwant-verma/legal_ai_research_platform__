import hashlib
import re
from dataclasses import dataclass
from app.core.config import get_settings
from app.models.legal import LegalChunk, LegalMetadata, PageText

settings = get_settings()

SECTION_RE = re.compile(
    r"^(?:SECTION\s+|Section\s+)?(?P<num>\d+[A-Z]?(?:\([0-9A-Za-z]+\))?)\.\s+(?P<title>.+)$"
)
ARTICLE_RE = re.compile(r"^(?:ARTICLE|Article)\s+(?P<num>\d+[A-Z]?)\b[.\-:]?\s*(?P<title>.*)$")
PARA_RE = re.compile(r"^\[?(?P<num>\d{1,4})\]?\.?\s+(?P<body>.+)$")
CHAPTER_RE = re.compile(r"^(CHAPTER|PART)\s+[IVXLC0-9A-Z-]+\b.*$", re.I)


@dataclass
class Line:
    page: int
    text: str


def estimate_tokens(text: str) -> int:
    # Deterministic tokenizer-independent estimate for storage/chunk control.
    return max(1, round(len(text) / 4))


def _chunk_id(document_id: str, seq: int, label: str | None) -> str:
    raw = f"{document_id}|{settings.chunk_version}|{seq}|{label or ''}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12].upper()
    return f"{document_id}_C{seq:05d}_{digest}"


def _to_lines(pages: list[PageText]) -> list[Line]:
    return [Line(page=p.page_number, text=line.strip()) for p in pages for line in p.text.splitlines() if line.strip()]


def _pack_units(
    document_id: str,
    metadata: LegalMetadata,
    units: list[tuple[list[Line], dict]],
) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []
    seq = 1
    max_tokens = settings.chunk_max_tokens

    for unit_lines, labels in units:
        if not unit_lines:
            continue
        # Split a very long legal unit while preserving label/provenance.
        current: list[Line] = []
        bounded_lines = []
        for line in unit_lines:
            cap = max_tokens * 4
            bounded_lines.extend(Line(line.page, line.text[i:i+cap]) for i in range(0, len(line.text), cap))
        for line in bounded_lines:
            candidate = "\n".join(x.text for x in current + [line])
            if current and estimate_tokens(candidate) > max_tokens:
                text = "\n".join(x.text for x in current)
                chunks.append(
                    LegalChunk(
                        chunk_id=_chunk_id(document_id, seq, labels.get("section") or labels.get("article") or labels.get("heading")),
                        document_id=document_id,
                        chunk_version=settings.chunk_version,
                        text=text,
                        page_start=min(x.page for x in current),
                        page_end=max(x.page for x in current),
                        part=labels.get("part"),
                        schedule=labels.get("schedule"),
                        section=labels.get("section"),
                        subsection=labels.get("subsection"),
                        article=labels.get("article"),
                        paragraph_ids=labels.get("paragraph_ids", []),
                        heading=labels.get("heading"),
                        token_estimate=estimate_tokens(text),
                        metadata=metadata,
                    )
                )
                seq += 1
                # Small line-level overlap to avoid severing context completely.
                overlap_chars = settings.chunk_overlap_tokens * 4
                carry: list[Line] = []
                used = 0
                for old in reversed(current):
                    if used + len(old.text) > overlap_chars:
                        break
                    carry.insert(0, old)
                    used += len(old.text)
                current = carry
            current.append(line)

        if current:
            text = "\n".join(x.text for x in current)
            chunks.append(
                LegalChunk(
                    chunk_id=_chunk_id(document_id, seq, labels.get("section") or labels.get("article") or labels.get("heading")),
                    document_id=document_id,
                    chunk_version=settings.chunk_version,
                    text=text,
                    page_start=min(x.page for x in current),
                    page_end=max(x.page for x in current),
                    part=labels.get("part"),
                    schedule=labels.get("schedule"),
                    section=labels.get("section"),
                    subsection=labels.get("subsection"),
                    article=labels.get("article"),
                    paragraph_ids=labels.get("paragraph_ids", []),
                    heading=labels.get("heading"),
                    token_estimate=estimate_tokens(text),
                    metadata=metadata,
                )
            )
            seq += 1
    return chunks


def _statute_units(lines: list[Line], constitution: bool = False) -> list[tuple[list[Line], dict]]:
    units: list[tuple[list[Line], dict]] = []
    current: list[Line] = []
    labels: dict = {}
    current_heading: str | None = None
    in_schedule = False
    seen_body = False
    current_part = None
    current_schedule = None

    def flush() -> None:
        nonlocal current, labels
        if current:
            units.append((current, {**labels,"part":current_part,"schedule":current_schedule}))
        current = []
        labels = {"heading": current_heading}

    for line in lines:
        if seen_body and re.match(r"^(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+SCHEDULE|SCHEDULE\b)", line.text, re.I):
            flush()
            in_schedule = True
            current_schedule = line.text
            current_heading = line.text
            current = [line]
            labels = {"heading": current_heading}
            continue
        if CHAPTER_RE.match(line.text):
            current_heading = line.text
            if line.text.upper().startswith("PART "):
                current_part = line.text
            if current:
                flush()
            current = [line]
            labels = {"heading": current_heading}
            continue

        article = ARTICLE_RE.match(line.text)
        section = SECTION_RE.match(line.text)
        # Footnote markers and entries in Schedules are retained as context, not provisions.
        footnote = bool(re.match(r"^\d+\.?\s*(?:Subs\.|Ins\.|Omitted|Added|The words|For the words)", line.text, re.I))
        if footnote or (constitution and in_schedule):
            article = section = None
        elif constitution and section:
            # Bare Article headings in official PDFs usually contain a title and an em dash.
            # Contents listings without operative text must not become Article labels.
            if re.search(r"[—–]|\.\s*[-―]|\b(?:shall|means|includes)\b", section.group('title')):
                article, section = section, None
            else:
                section = None
        if article:
            seen_body = True
            flush()
            labels = {"article": article.group("num"), "heading": article.group("title") or current_heading}
            current = [line]
        elif section:
            flush()
            labels = {"section": section.group("num"), "heading": section.group("title") or current_heading}
            current = [line]
        else:
            current.append(line)

    flush()
    return units


def _judgment_units(lines: list[Line]) -> list[tuple[list[Line], dict]]:
    # Group numbered judgment paragraphs, then pack several related paragraphs
    # until target size. Paragraph IDs are preserved for citation traceability.
    paragraphs: list[tuple[list[Line], str | None]] = []
    current: list[Line] = []
    para_id: str | None = None

    def flush_para() -> None:
        nonlocal current, para_id
        if current:
            paragraphs.append((current, para_id))
        current = []
        para_id = None

    for line in lines:
        m = PARA_RE.match(line.text)
        if m and len(line.text) > 15:
            flush_para()
            para_id = m.group("num")
            current = [line]
        else:
            current.append(line)
    flush_para()

    units: list[tuple[list[Line], dict]] = []
    buf: list[Line] = []
    ids: list[str] = []
    target = settings.chunk_target_tokens

    for p_lines, p_id in paragraphs:
        candidate = "\n".join(x.text for x in buf + p_lines)
        if buf and estimate_tokens(candidate) > target:
            units.append((buf, {"paragraph_ids": ids.copy()}))
            buf, ids = [], []
        buf.extend(p_lines)
        if p_id:
            ids.append(p_id)

    if buf:
        units.append((buf, {"paragraph_ids": ids.copy()}))
    return units


def _generic_units(lines: list[Line]) -> list[tuple[list[Line], dict]]:
    units: list[tuple[list[Line], dict]] = []
    buf: list[Line] = []
    for line in lines:
        candidate = "\n".join(x.text for x in buf + [line])
        if buf and estimate_tokens(candidate) > settings.chunk_target_tokens:
            units.append((buf, {}))
            buf = []
        buf.append(line)
    if buf:
        units.append((buf, {}))
    return units


def _join_headings(lines):
    joined=[];i=0
    while i<len(lines):
        line=lines[i];text=line.text
        if SECTION_RE.match(text) and not re.search(r'[—–]',text):
            j=i+1;parts=[text]
            while j<min(i+4,len(lines)) and lines[j].page==line.page:
                if SECTION_RE.match(lines[j].text) or CHAPTER_RE.match(lines[j].text): break
                parts.append(lines[j].text)
                if re.search(r'[—–]',lines[j].text):
                    text=' '.join(parts);i=j;break
                j+=1
        joined.append(Line(line.page,text));i+=1
    return joined

def chunk_document(document_id: str, pages: list[PageText], metadata: LegalMetadata) -> list[LegalChunk]:
    lines = _to_lines(pages)
    if metadata.document_type == "constitution" or "constitution of india" in (metadata.title or "").lower():
        lines = _join_headings(lines)
    if metadata.document_type in {"constitution", "statute", "rule", "regulation"}:
        units = _statute_units(lines, constitution=metadata.document_type == "constitution" or "constitution of india" in (metadata.title or "").lower())
    elif metadata.document_type == "judgment":
        units = _judgment_units(lines)
    else:
        units = _generic_units(lines)
    return _pack_units(document_id, metadata, units)
