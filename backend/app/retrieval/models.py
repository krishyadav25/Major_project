"""Data models for retrieval operations.

These Pydantic models define query representations, retrieved search results,
and fusion configurations used across dense, sparse, and hybrid search stages.
"""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single retrieved chunk with citation metadata and relevance scores."""

    chunk_id: str = Field(
        description="Unique chunk identifier '{doc_id}::chunk::{chunk_index:04d}'."
    )
    doc_id: str = Field(description="Stable identifier of the source document.")
    filename: str = Field(description="Source PDF filename.")
    page_number: int = Field(ge=1, description="1-based page number where chunk originates.")
    chunk_index: int = Field(ge=0, description="0-based sequence index of the chunk.")
    text: str = Field(description="Cleaned chunk text.")
    scheme_name: str | None = Field(default=None, description="Government scheme name if known.")

    score: float = Field(
        default=0.0, description="Primary relevance score (e.g. RRF or raw score)."
    )
    dense_score: float | None = Field(
        default=None, description="Similarity score from dense vector search."
    )
    sparse_score: float | None = Field(
        default=None, description="Relevance score from BM25 sparse search."
    )
    dense_rank: int | None = Field(
        default=None, ge=1, description="1-based rank from dense retrieval."
    )
    sparse_rank: int | None = Field(
        default=None, ge=1, description="1-based rank from sparse retrieval."
    )
    rrf_score: float | None = Field(
        default=None, ge=0.0, description="Reciprocal Rank Fusion combined score."
    )

    @property
    def source_citation(self) -> str:
        """Format a human-readable citation for evidence verification."""
        prefix = f"[{self.scheme_name}] " if self.scheme_name else ""
        return f"{prefix}{self.filename} (Page {self.page_number})"


class RetrievalQuery(BaseModel):
    """User retrieval query parameters."""

    query: str = Field(description="Search query string.")
    scheme_name: str | None = Field(default=None, description="Optional scheme filter.")
    top_k: int = Field(default=5, ge=1, description="Target number of results to return.")
    rrf_k: int = Field(
        default=60, ge=1, description="Smoothing constant for Reciprocal Rank Fusion."
    )


class FusionConfig(BaseModel):
    """Configuration for result fusion."""

    rrf_k: int = Field(default=60, ge=1, description="RRF constant k (default 60).")
    top_k: int = Field(default=5, ge=1, description="Number of top results to return.")
