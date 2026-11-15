
#!/usr/bin/env python3
"""
Extract text from arXiv PDFs using PyMuPDF4LLM.

Install:
    pip install pymupdf4llm

Example (PowerShell):
    python scripts/extract_arxiv_text.py --pdf-dir dataset/pdfs --output-dir dataset/extracted --manifest dataset/dataset_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pymupdf4llm
import pymupdf


def normalize_extracted_text(text: str) -> str:
    """Normalize whitespace while preserving Markdown structure."""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove standalone page numbers.
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # Remove trailing whitespace from lines.
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(pdf_path: Path) -> tuple[str, int]:
    """Extract PDF content as Markdown using PyMuPDF4LLM."""

    with pymupdf.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError("PDF is encrypted")

        page_count = len(document)

        if page_count == 0:
            raise ValueError("PDF contains no pages")

        # Extract the entire document as Markdown.
        markdown = pymupdf4llm.to_markdown(
            document,
            show_progress=False,
        )

    markdown = normalize_extracted_text(markdown)

    return markdown + "\n", page_count


def update_manifest(
    manifest_path: Path,
    results: dict[str, dict[str, str]]
) -> None:
    """Update extraction status and output details in the existing CSV manifest."""

    if not manifest_path.exists():
        print(
            f"Warning: manifest not found: {manifest_path}",
            file=sys.stderr
        )
        return

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        print(
            "Warning: manifest has no header; skipping update.",
            file=sys.stderr
        )
        return

    for field in (
        "extracted_path",
        "page_count",
        "extraction_error"
    ):
        if field not in fieldnames:
            fieldnames.append(field)

    updated = 0

    for row in rows:
        paper_id = str(row.get("arxiv_id", "")).strip()
        result = results.get(paper_id)

        if result is None:
            continue

        row["extraction_status"] = result["status"]
        row["extracted_path"] = result["extracted_path"]
        row["page_count"] = result["page_count"]
        row["extraction_error"] = result["error"]

        updated += 1

    temp_path = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(manifest_path)

    print(f"Updated {updated} manifest entries.")


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Extract text from arXiv PDFs using PyMuPDF4LLM."
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional dataset_manifest.csv"
    )

    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        parser.error(
            f"PDF directory does not exist: {args.pdf_dir}"
        )

    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {args.pdf_dir}")
        return 1

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}
    successes = 0
    failures = 0

    for index, pdf_path in enumerate(pdf_files, start=1):

        paper_id = pdf_path.stem

        # Save extracted Markdown instead of plain text.
        output_path = args.output_dir / f"{paper_id}.md"

        print(
            f"[{index}/{len(pdf_files)}] {pdf_path.name}"
        )

        try:
            markdown, page_count = extract_pdf(pdf_path)

            # Detect PDFs that open but contain almost no extractable text.
            check_text = re.sub(
                r"!\[[^\]]*\]\([^)]*\)",
                "",
                markdown
            ).strip()

            if len(check_text) < 100:
                raise ValueError(
                    "Very little extractable text found; this may be a "
                    "scanned PDF requiring OCR."
                )

            output_path.write_text(
                markdown,
                encoding="utf-8"
            )

            results[paper_id] = {
                "status": "success",
                "extracted_path": str(output_path),
                "page_count": str(page_count),
                "error": "",
            }

            successes += 1

            print(
                f"  Extracted {page_count} pages, "
                f"{len(check_text):,} characters."
            )

        except Exception as exc:

            results[paper_id] = {
                "status": "failed",
                "extracted_path": "",
                "page_count": "",
                "error": str(exc),
            }

            failures += 1

            print(
                f"  ERROR: {exc}",
                file=sys.stderr
            )

    if args.manifest:
        update_manifest(args.manifest, results)

    print("\nExtraction finished.")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Output directory: {args.output_dir}")

    if failures:
        print("Review failed PDFs. Scanned PDFs may require OCR.")

    print(
        "\nNote: This extracts the entire paper, including its abstract. "
        "Keep the abstract in the extracted file for now; exclude it when "
        "preparing the separate model-input text."
    )

    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
