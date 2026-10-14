"""Batch ingest a directory of PDFs with resume-by-hash behavior.

Usage:
    python scripts/ingest_folder.py /path/to/pdfs --report ingestion_report.jsonl

If `file.pdf` has `file.metadata.json` next to it, the legal metadata sidecar is used.
Bad/scanned PDFs are not silently indexed; they are marked OCR-required.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.mongo import ensure_indexes  # noqa: E402
from app.services.ingestion import ingest_pdf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--report", type=Path, default=Path("ingestion_report.jsonl"))
    args = parser.parse_args()

    ensure_indexes()
    pdfs = sorted(args.folder.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")

    with args.report.open("a", encoding="utf-8") as report:
        for index, pdf in enumerate(pdfs, start=1):
            try:
                result = ingest_pdf(pdf)
            except Exception as exc:
                result = {"status": "failed", "file": str(pdf), "error": repr(exc)}
            result["file"] = str(pdf)
            report.write(json.dumps(result, ensure_ascii=False) + "\n")
            report.flush()
            print(f"[{index}/{len(pdfs)}] {pdf.name}: {result.get('status')}")


if __name__ == "__main__":
    main()
