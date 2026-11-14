#!/usr/bin/env python3
"""
Select a small, category-balanced sample from the Kaggle arXiv metadata file,
download the corresponding arXiv PDFs, and save abstracts and a CSV manifest.

Expected metadata format: JSON Lines (one JSON object per line), as used by
the common arXiv Kaggle metadata snapshot.

Example:
    python prepare_arxiv_dataset.py dataset/metadata/arxiv-metadata-oai-snapshot.json `
        --output-dir dataset `
        --papers 25 `
        --categories cs.CL cs.AI cs.LG cs.CV

Dependency:
    pip install requests
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


DEFAULT_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "cs.CV"]
ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}"
USER_AGENT = "StudentResearchDatasetDownloader/1.0 (academic project; contact: replace-with-your-email)"


def clean_filename(value: str) -> str:
    """Make a safe filename component."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._") or "paper"


def parse_categories(value: str) -> list[str]:
    """Return the category tags in an arXiv metadata record."""
    return value.split()


def primary_category(categories: list[str], requested: list[str]) -> str | None:
    """Assign a paper to the first requested category it belongs to."""
    category_set = set(categories)
    for category in requested:
        if category in category_set:
            return category
    return None


def select_papers(
    metadata_path: Path,
    requested_categories: list[str],
    number_to_select: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Perform category-stratified reservoir sampling without loading the entire
    1.7-million-record metadata file into memory.
    """
    rng = random.Random(seed)
    per_category_target = (number_to_select + len(requested_categories) - 1) // len(
        requested_categories
    )
    reservoirs: dict[str, list[dict[str, Any]]] = {
        category: [] for category in requested_categories
    }
    seen: dict[str, int] = defaultdict(int)

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping invalid JSON on line {line_number}",
                    file=sys.stderr,
                )
                continue

            paper_id = str(record.get("id", "")).strip()
            title = " ".join(str(record.get("title", "")).split())
            abstract = " ".join(str(record.get("abstract", "")).split())
            categories = parse_categories(str(record.get("categories", "")))

            if not paper_id or not title or not abstract:
                continue

            category = primary_category(categories, requested_categories)
            if category is None:
                continue

            # Skip records explicitly marked as withdrawn, if that field exists.
            if record.get("withdrawn") or record.get("is_withdrawn"):
                continue

            seen[category] += 1
            candidate = {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "categories": categories,
                "selected_category": category,
                "authors": record.get("authors", ""),
                "submitter": record.get("submitter", ""),
                "doi": record.get("doi", ""),
                "journal-ref": record.get("journal-ref", ""),
                "comments": record.get("comments", ""),
                "versions": record.get("versions", []),
            }

            reservoir = reservoirs[category]
            if len(reservoir) < per_category_target:
                reservoir.append(candidate)
            else:
                replacement_index = rng.randrange(seen[category])
                if replacement_index < per_category_target:
                    reservoir[replacement_index] = candidate

    selected = [
        paper
        for category in requested_categories
        for paper in reservoirs[category]
    ]
    rng.shuffle(selected)
    selected = selected[:number_to_select]

    counts = defaultdict(int)
    for paper in selected:
        counts[paper["selected_category"]] += 1

    print(f"Selected {len(selected)} papers from {metadata_path}")
    for category in requested_categories:
        print(
            f"  {category}: {counts[category]} selected "
            f"({seen[category]} eligible records scanned)"
        )

    if len(selected) < number_to_select:
        print(
            f"Warning: requested {number_to_select}, but only found "
            f"{len(selected)} eligible papers in the selected categories.",
            file=sys.stderr,
        )

    return selected


def download_pdf(
    session: requests.Session,
    paper_id: str,
    destination: Path,
    timeout: int,
    retries: int,
) -> tuple[str, str]:
    """Download one PDF, validating that the response looks like a PDF."""
    url = ARXIV_PDF_URL.format(paper_id=paper_id)
    temporary_path = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            content = response.content
            if not content.startswith(b"%PDF-"):
                raise ValueError(
                    f"Response does not look like a PDF "
                    f"(Content-Type: {content_type or 'unknown'})"
                )

            temporary_path.write_bytes(content)
            temporary_path.replace(destination)
            return "success", ""

        except (requests.RequestException, ValueError, OSError) as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            message = str(exc)
            if attempt < retries:
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"  Attempt {attempt}/{retries} failed for {paper_id}: "
                    f"{message}. Retrying in {wait_seconds}s."
                )
                time.sleep(wait_seconds)
            else:
                return "failed", message

    return "failed", "Unknown download error"


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select arXiv papers from metadata and download their PDFs."
    )
    parser.add_argument(
        "metadata",
        type=Path,
        help="Path to the Kaggle arXiv JSON Lines metadata file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/pilot"),
        help="Output directory (default: dataset/pilot).",
    )
    parser.add_argument(
        "--papers",
        type=int,
        default=25,
        help="Number of papers to select (default: 25).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="arXiv categories to sample from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection (default: 42).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between PDF requests (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Download timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum download attempts per paper (default: 3).",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="",
        help="Contact email to include in the downloader User-Agent.",
    )
    args = parser.parse_args()

    if not args.metadata.is_file():
        parser.error(f"Metadata file not found: {args.metadata}")
    if args.papers < 1:
        parser.error("--papers must be at least 1")
    if not args.categories:
        parser.error("Provide at least one category with --categories")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    output_dir = args.output_dir
    pdf_dir = output_dir / "pdfs"
    abstract_dir = output_dir / "abstracts"
    metadata_dir = output_dir / "metadata"

    for directory in (pdf_dir, abstract_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected = select_papers(
        metadata_path=args.metadata,
        requested_categories=args.categories,
        number_to_select=args.papers,
        seed=args.seed,
    )
    if not selected:
        print("No matching papers found. Check the metadata and categories.")
        return 1

    write_json(metadata_dir / "selected_papers.json", selected)

    user_agent = USER_AGENT
    if args.email:
        user_agent = user_agent.replace(
            "replace-with-your-email", args.email
        )
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    manifest_rows: list[dict[str, str]] = []

    for index, paper in enumerate(selected, start=1):
        paper_id = paper["id"]
        safe_id = clean_filename(paper_id)
        pdf_path = pdf_dir / f"{safe_id}.pdf"
        abstract_path = abstract_dir / f"{safe_id}.txt"

        # Save the reference abstract separately; do not feed it to the model.
        abstract_path.write_text(paper["abstract"], encoding="utf-8")

        print(f"[{index}/{len(selected)}] Downloading {paper_id}: {paper['title']}")
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            pdf_status, download_error = "already_exists", ""
        else:
            pdf_status, download_error = download_pdf(
                session=session,
                paper_id=paper_id,
                destination=pdf_path,
                timeout=args.timeout,
                retries=args.retries,
            )

        manifest_rows.append(
            {
                "arxiv_id": paper_id,
                "title": paper["title"],
                "categories": " ".join(paper["categories"]),
                "selected_category": paper["selected_category"],
                "pdf_url": ARXIV_PDF_URL.format(paper_id=paper_id),
                "pdf_path": str(pdf_path) if pdf_status in {"success", "already_exists"} else "",
                "abstract_path": str(abstract_path),
                "pdf_status": pdf_status,
                "extraction_status": "not_started",
                "token_count": "",
                "download_error": download_error,
            }
        )

        if index < len(selected):
            time.sleep(args.delay)

    manifest_path = output_dir / "dataset_manifest.csv"
    fieldnames = [
        "arxiv_id",
        "title",
        "categories",
        "selected_category",
        "pdf_url",
        "pdf_path",
        "abstract_path",
        "pdf_status",
        "extraction_status",
        "token_count",
        "download_error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    successful = sum(
        row["pdf_status"] in {"success", "already_exists"}
        for row in manifest_rows
    )
    print("\nFinished.")
    print(f"PDFs available: {successful}/{len(selected)}")
    print(f"Manifest: {manifest_path}")
    print(f"Selected metadata: {metadata_dir / 'selected_papers.json'}")
    print(f"PDF directory: {pdf_dir}")
    print(f"Abstract directory: {abstract_dir}")
    print(
        "\nNext step: extract PDF text and record each paper's LED tokenizer "
        "token count. The abstracts are reference summaries and must be "
        "excluded from model input."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
