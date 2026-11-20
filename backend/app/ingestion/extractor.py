"""PDF text extraction using PyMuPDF (fitz).

This module loads a PDF file and extracts raw text from every page.
It does NOT perform OCR. Pages with no extractable text layer are
flagged with ``is_empty=True`` and ``is_scanned`` is set on the
document metadata when the entire document appears to be image-only.

Supported text: UTF-8, including Hindi (Devanagari) and English.
"""

from pathlib import Path

import pymupdf as fitz  # PyMuPDF ≥ 1.24 recommends `import pymupdf`

from app.ingestion.models import (
    DocumentMetadata,
    ExtractedPage,
    ProcessedDocument,
    compute_content_hash,
    make_doc_id,
)

# A page is considered "empty" if it has fewer characters than this threshold
# after stripping whitespace.  Accounts for stray whitespace-only blocks.
_EMPTY_PAGE_THRESHOLD = 10


def extract_pdf(
    filepath: str | Path,
    scheme_name: str | None = None,
    use_content_hash: bool = False,
) -> ProcessedDocument:
    """Load a PDF and extract text from every page.

    Args:
        filepath: Path to the PDF file.
        scheme_name: Optional name of the government scheme this PDF describes.

    Returns:
        A :class:`ProcessedDocument` containing per-page raw text and metadata.

    Raises:
        FileNotFoundError: When the file does not exist at the given path.
        ValueError: When the path is not a file, is not a PDF, or contains zero pages.
        RuntimeError: When PyMuPDF cannot open or parse the file.
    """
    path = Path(filepath).resolve()

    # --- Path validation ---
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix!r}")

    # --- Open and validate the PDF ---
    try:
        doc: fitz.Document = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF could not open '{path}': {exc}") from exc

    with doc:
        if doc.is_encrypted:
            raise RuntimeError(
                f"PDF '{path.name}' is password-protected and cannot be read without a key."
            )

        page_count = len(doc)
        if page_count == 0:
            raise ValueError(f"PDF contains no pages: '{path.name}'")

        content_hash = compute_content_hash(path.read_bytes())
        doc_id = make_doc_id(
            path,
            prefix=scheme_name,
            content_hash=content_hash if use_content_hash else None,
        )
        pages: list[ExtractedPage] = []

        for page_index in range(page_count):
            page: fitz.Page = doc[page_index]
            raw_text: str = page.get_text("text")  # UTF-8 plain text
            is_empty = len(raw_text.strip()) < _EMPTY_PAGE_THRESHOLD

            pages.append(
                ExtractedPage(
                    page_number=page_index + 1,  # 1-based
                    raw_text=raw_text,
                    char_count=len(raw_text),
                    is_empty=is_empty,
                )
            )

    # Mark the document as scanned when NO page has extractable text.
    # Individual empty pages within an otherwise text-bearing PDF are normal
    # (title pages, diagrams) and do NOT set is_scanned.
    is_scanned = page_count > 0 and all(p.is_empty for p in pages)

    metadata = DocumentMetadata(
        doc_id=doc_id,
        filename=path.name,
        filepath=str(path),
        page_count=page_count,
        is_scanned=is_scanned,
        scheme_name=scheme_name,
        content_hash=content_hash,
    )

    return ProcessedDocument(metadata=metadata, pages=pages)
