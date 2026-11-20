"""Data models for the ingestion pipeline.

These Pydantic models represent the structured output at each stage of
the pipeline: raw extraction → cleaning → chunking.
"""

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


def make_doc_id(filepath: str | Path) -> str:
    """Derive a stable document ID from a file path.

    Uses the filename stem (without extension) in lower-snake-case.

    Example:
        ``PM_Kisan_Guidelines.pdf`` → ``pm_kisan_guidelines``
    """
    stem = Path(filepath).stem
    return stem.lower().replace(" ", "_").replace("-", "_")
