"""Data models for the ingestion pipeline.

These Pydantic models represent the structured output at each stage of
the pipeline: raw extraction → cleaning → chunking.
"""

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata about a source PDF document."""

    doc_id: str = Field(description="Stable identifier derived from the source filename.")
    filename: str = Field(description="Original filename of the PDF, including extension.")
    filepath: str = Field(description="Absolute path to the PDF file at extraction time.")
    page_count: int = Field(ge=0, description="Total number of pages in the document.")
    is_scanned: bool = Field(
        default=False,
        description=(
            "True when the PDF contains no extractable text layer. "
            "Scanned documents require OCR processing (not yet supported)."
        ),
    )
    scheme_name: str | None = Field(
        default=None,
        description="Name of the government scheme this document describes, if known.",
    )
    content_hash: str | None = Field(
        default=None,
        description="Deterministic content hash of the source document.",
    )

    @field_validator("doc_id")
    @classmethod
    def doc_id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("doc_id must not be empty.")
        return v


class ExtractedPage(BaseModel):
    """Raw text extracted from a single PDF page."""

    page_number: int = Field(ge=1, description="1-based page number within the source document.")
    raw_text: str = Field(description="Unmodified text as extracted by PyMuPDF.")
    char_count: int = Field(ge=0, description="Number of characters in raw_text.")
    is_empty: bool = Field(description="True when the page contains no extractable text.")


class ProcessedDocument(BaseModel):
    """A PDF document after extraction and text cleaning."""

    metadata: DocumentMetadata
    pages: list[ExtractedPage] = Field(default_factory=list)
    cleaned_pages: list[str] = Field(
        default_factory=list,
        description="Cleaned text for each page, parallel to `pages`.",
    )

    @property
    def total_chars(self) -> int:
        """Total character count across all extracted pages."""
        return sum(p.char_count for p in self.pages)

    @property
    def non_empty_page_count(self) -> int:
        """Number of pages that contain extractable text."""
        return sum(1 for p in self.pages if not p.is_empty)


class ChunkMetadata(BaseModel):
    """Metadata attached to a single text chunk."""

    chunk_id: str = Field(description="Stable chunk identifier: '{doc_id}::chunk::{index:04d}'.")
    doc_id: str = Field(description="Identifier of the source document.")
    filename: str = Field(description="Source filename for traceability.")
    page_number: int = Field(ge=1, description="1-based page number where this chunk originates.")
    chunk_index: int = Field(
        ge=0, description="0-based position of this chunk within the document."
    )
    char_count: int = Field(ge=0, description="Character count of the chunk text.")
    scheme_name: str | None = Field(
        default=None,
        description="Scheme name from the source document, if known.",
    )


class TextChunk(BaseModel):
    """A self-contained piece of text ready for embedding and retrieval."""

    text: str = Field(description="The cleaned chunk text.")
    metadata: ChunkMetadata

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("TextChunk text must not be empty.")
        return v


def compute_content_hash(content: bytes | str) -> str:
    """Compute a deterministic 8-character SHA-256 hash of document content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:8]


def make_doc_id(
    filepath: str | Path,
    prefix: str | None = None,
    content_hash: str | None = None,
) -> str:
    """Derive a stable, collision-resistant document ID from a file path.

    Sanitizes the filename stem to contain only lowercase alphanumeric
    characters and single underscores. Punctuation (such as parentheses,
    brackets, dots, and symbols) is replaced with underscores.

    If an optional content hash is supplied, it is appended to disambiguate
    different files with identical filenames.

    If an optional prefix (e.g. scheme name) is supplied and not already
    present in the stem, it is prepended to namespace common filenames like
    ``guidelines.pdf`` across different schemes.

    Args:
        filepath: Path to the document or filename string.
        prefix: Optional namespace prefix (e.g. scheme name).
        content_hash: Optional content hash (e.g. SHA-256 prefix).

    Example:
        ``PM_Kisan (2024) [v1.0].pdf`` → ``pm_kisan_2024_v1_0``
        ``guidelines.pdf`` with prefix ``PM-Kisan`` → ``pm_kisan_guidelines``
        ``guidelines.pdf`` with hash ``e3b0c442`` → ``guidelines_e3b0c442``
    """
    stem = Path(filepath).stem
    # Replace all non-alphanumeric characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    if not sanitized:
        sanitized = "doc"

    if content_hash and content_hash.strip():
        sanitized_hash = re.sub(r"[^a-zA-Z0-9]+", "", content_hash)[:8].lower()
        if sanitized_hash and not sanitized.endswith(sanitized_hash):
            sanitized = f"{sanitized}_{sanitized_hash}"

    if prefix and prefix.strip():
        sanitized_prefix = re.sub(r"[^a-zA-Z0-9]+", "_", prefix).strip("_").lower()
        if sanitized_prefix and not sanitized.startswith(sanitized_prefix):
            return f"{sanitized_prefix}_{sanitized}"

    return sanitized
