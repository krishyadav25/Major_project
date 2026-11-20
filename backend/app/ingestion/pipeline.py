"""End-to-end document ingestion pipeline.

Orchestrates the three ingestion stages in order:
    1. :func:`~app.ingestion.extractor.extract_pdf` — load PDF, extract raw text.
    2. :func:`~app.ingestion.cleaner.clean_document_pages` — normalize text.
    3. :func:`~app.ingestion.chunker.chunk_document` — split into TextChunks.

Usage example (from another backend module)::

    from pathlib import Path
    from app.ingestion.pipeline import ingest_pdf
    from app.ingestion.chunker import ChunkConfig

    chunks = ingest_pdf(
        filepath=Path("data/raw/pm_kisan_guidelines.pdf"),
        scheme_name="PM-Kisan",
        config=ChunkConfig(chunk_size=512, chunk_overlap=64),
    )
    for chunk in chunks:
        print(chunk.metadata.chunk_id, chunk.text[:80])
"""

from pathlib import Path

from app.core.logging import get_logger
from app.ingestion.chunker import ChunkConfig, chunk_document
from app.ingestion.cleaner import clean_document_pages
from app.ingestion.extractor import extract_pdf
from app.ingestion.models import ProcessedDocument, TextChunk

logger = get_logger(__name__)


def ingest_pdf(
    filepath: str | Path,
    scheme_name: str | None = None,
    config: ChunkConfig | None = None,
    use_content_hash: bool = False,
) -> list[TextChunk]:
    """Extract, clean and chunk a single PDF document.

    This is the primary entry-point for the ingestion pipeline.  Downstream
    modules (retrieval, evaluation) should call this function rather than
    the individual stage functions.

    Args:
        filepath: Path to the PDF file to ingest.
        scheme_name: Optional name of the government scheme this PDF covers.
            Stored in every chunk's metadata for filtering and traceability.
        config: Chunking configuration.  Uses sensible defaults when omitted.

    Returns:
        Ordered list of :class:`~app.ingestion.models.TextChunk` objects
        ready for embedding and indexing.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When the path is not a PDF, or contains zero pages.
        RuntimeError: When PyMuPDF cannot open the file, or when the document
            is a scanned-only PDF with no extractable text layer.
    """
    path = Path(filepath)
    logger.info("Ingestion started: '%s'", path.name)

    # Stage 1 — Extract
    document: ProcessedDocument = extract_pdf(
        path, scheme_name=scheme_name, use_content_hash=use_content_hash
    )

    if document.metadata.is_scanned:
        raise RuntimeError(
            f"'{path.name}' appears to be a scanned PDF with no extractable text layer. "
            "OCR support is planned for a future milestone."
        )

    logger.info(
        "Extracted %d page(s) from '%s' (%d non-empty).",
        document.metadata.page_count,
        path.name,
        document.non_empty_page_count,
    )

    # Stage 2 — Clean
    document.cleaned_pages = clean_document_pages([page.raw_text for page in document.pages])

    # Stage 3 — Chunk
    chunks = chunk_document(document, config=config)

    logger.info(
        "Ingestion complete: '%s' → %d chunk(s).",
        path.name,
        len(chunks),
    )
    return chunks
