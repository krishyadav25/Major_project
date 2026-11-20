"""Configurable, page-aware text chunking.

Splits cleaned page text into overlapping chunks while preserving:
- Source page number for every chunk.
- Document and filename metadata.
- Stable, deterministic chunk IDs.
- Word boundaries (no word is split across chunks).
- Short-document and long-paragraph edge cases.

Design decisions:
- Chunking happens per-page so a chunk never silently spans two pages.
  This keeps source traceability exact.
- Overlap is applied at the word level to maintain context continuity
  between adjacent chunks of the same page.
- Empty chunks (after stripping) are discarded.
"""

from dataclasses import dataclass

from app.ingestion.models import ChunkMetadata, DocumentMetadata, ProcessedDocument, TextChunk

# Default parameters match common RAG practice for dense retrieval.
DEFAULT_CHUNK_SIZE = 512  # characters
DEFAULT_CHUNK_OVERLAP = 64  # characters


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking configuration parameters.

    Attributes:
        chunk_size: Target maximum character length for a chunk.
        chunk_overlap: Number of characters of overlap between consecutive chunks
            on the same page.  Must be strictly less than ``chunk_size``.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})."
            )


def _chunk_text_into_pieces(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a string into overlapping character-capped pieces at word boundaries.

    Args:
        text: The cleaned text to split.
        chunk_size: Maximum character length for each piece.
        overlap: Character count of overlap between consecutive pieces.

    Returns:
        A list of non-empty text pieces.  Returns ``[]`` for blank input.
    """
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0  # index into the word list

    while start < len(words):
        # Greedily add words until we exceed chunk_size
        end = start
        current_length = 0
        while end < len(words):
            word_len = len(words[end]) + (1 if end > start else 0)  # +1 for space
            if current_length + word_len > chunk_size and end > start:
                break
            current_length += word_len
            end += 1

        piece = " ".join(words[start:end])
        if piece.strip():
            chunks.append(piece)

        if end >= len(words):
            break

        # Determine the next start position respecting overlap.
        # We step back enough words to cover ~overlap characters.
        overlap_chars = 0
        step_back = 0
        for i in range(end - 1, start - 1, -1):
            overlap_chars += len(words[i]) + 1
            step_back += 1
            if overlap_chars >= overlap:
                break

        next_start = end - step_back
        # Safety: always advance at least one word to prevent infinite loops.
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def chunk_document(
    document: ProcessedDocument,
    config: ChunkConfig | None = None,
) -> list[TextChunk]:
    """Chunk a processed document into TextChunk objects.

    Each chunk retains the page number it came from.  Chunks never
    straddle page boundaries.  Empty pages produce no chunks.

    Args:
        document: A :class:`ProcessedDocument` after extraction and cleaning.
        config: Chunking configuration.  Uses :class:`ChunkConfig` defaults
            when not supplied.

    Returns:
        Ordered list of :class:`TextChunk` objects.
    """
    if config is None:
        config = ChunkConfig()

    metadata: DocumentMetadata = document.metadata
    chunks: list[TextChunk] = []
    chunk_index = 0

    for page, cleaned_text in zip(document.pages, document.cleaned_pages, strict=True):
        if not cleaned_text.strip():
            continue  # Skip empty pages — they produce no chunks

        pieces = _chunk_text_into_pieces(cleaned_text, config.chunk_size, config.chunk_overlap)
        for piece in pieces:
            chunk_id = f"{metadata.doc_id}::chunk::{chunk_index:04d}"
            chunk_meta = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=metadata.doc_id,
                filename=metadata.filename,
                page_number=page.page_number,
                chunk_index=chunk_index,
                char_count=len(piece),
                scheme_name=metadata.scheme_name,
            )
            chunks.append(TextChunk(text=piece, metadata=chunk_meta))
            chunk_index += 1

    return chunks
