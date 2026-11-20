"""BM25 sparse keyword retrieval using rank-bm25.

Provides keyword-based lexical retrieval with:
- Multilingual tokenization (English, Hindi, Hinglish, numbers, currency)
- Scheme-based metadata pre-filtering
- Safe, human-readable JSON persistence (zero pickle vulnerability)
- Idempotent indexing by chunk ID
"""

import json
from collections.abc import Callable
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.models import ChunkMetadata, TextChunk
from app.retrieval.models import SearchResult
from app.retrieval.tokenization import tokenize_multilingual


class BM25Retriever:
    """Sparse keyword retriever backed by BM25Okapi."""

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] = tokenize_multilingual,
    ) -> None:
        self.tokenizer = tokenizer
        self.chunks: list[TextChunk] = []
        self._chunk_map: dict[str, TextChunk] = {}
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def index_chunks(self, chunks: list[TextChunk]) -> int:
        """Index or update text chunks in the BM25 model.

        Deduplicates by chunk_id so re-indexing the same document or chunk
        updates existing entries deterministically.

        Args:
            chunks: List of TextChunk objects to index.

        Returns:
            Total count of indexed chunks.
        """
        if not chunks:
            return len(self.chunks)

        for chunk in chunks:
            self._chunk_map[chunk.metadata.chunk_id] = chunk

        self.chunks = list(self._chunk_map.values())
        self.tokenized_corpus = [self.tokenizer(c.text) for c in self.chunks]

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

        return len(self.chunks)

    def delete_document(self, doc_id: str) -> int:
        """Remove all indexed chunks belonging to the given doc_id.

        Rebuilds the BM25 model after deletion.

        Args:
            doc_id: Identifier of the document whose chunks should be removed.

        Returns:
            Total count of remaining indexed chunks.
        """
        keys_to_remove = [cid for cid, c in self._chunk_map.items() if c.metadata.doc_id == doc_id]
        if not keys_to_remove:
            return len(self.chunks)

        for cid in keys_to_remove:
            del self._chunk_map[cid]

        self.chunks = list(self._chunk_map.values())
        self.tokenized_corpus = [self.tokenizer(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        scheme_name: str | None = None,
    ) -> list[SearchResult]:
        """Perform BM25 search for a given query string.

        Args:
            query: User search query.
            top_k: Maximum number of ranked results to return.
            scheme_name: Optional government scheme name to filter by.

        Returns:
            List of SearchResult objects ordered by relevance score descending.
        """
        if not query.strip() or self.bm25 is None or not self.chunks:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        candidates: list[tuple[int, float, TextChunk]] = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores, strict=True)):
            if score <= 0.0:
                continue
            if scheme_name and chunk.metadata.scheme_name != scheme_name:
                continue
            candidates.append((idx, float(score), chunk))

        # Sort descending by score; break ties deterministically by chunk_id
        candidates.sort(key=lambda item: (-item[1], item[2].metadata.chunk_id))
        top_candidates = candidates[:top_k]

        results: list[SearchResult] = []
        for rank, (_idx, score, chunk) in enumerate(top_candidates, start=1):
            results.append(
                SearchResult(
                    chunk_id=chunk.metadata.chunk_id,
                    doc_id=chunk.metadata.doc_id,
                    filename=chunk.metadata.filename,
                    page_number=chunk.metadata.page_number,
                    chunk_index=chunk.metadata.chunk_index,
                    text=chunk.text,
                    scheme_name=chunk.metadata.scheme_name,
                    score=score,
                    sparse_score=score,
                    sparse_rank=rank,
                )
            )

        return results

    def save_to_json(self, filepath: str | Path) -> None:
        """Serialize indexed chunks to a safe, portable JSON file.

        Does NOT use pickle to prevent deserialization security vulnerabilities.

        Args:
            filepath: Destination JSON file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "chunk_count": len(self.chunks),
            "chunks": [
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata.model_dump(),
                }
                for chunk in self.chunks
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, filepath: str | Path) -> int:
        """Load and reconstruct the BM25 index from a JSON file.

        Args:
            filepath: Source JSON file path.

        Returns:
            Count of loaded and indexed chunks.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks: list[TextChunk] = []
        for item in data.get("chunks", []):
            meta = ChunkMetadata(**item["metadata"])
            chunks.append(TextChunk(text=item["text"], metadata=meta))

        return self.index_chunks(chunks)

    @classmethod
    def from_json(
        cls,
        filepath: str | Path,
        tokenizer: Callable[[str], list[str]] = tokenize_multilingual,
    ) -> "BM25Retriever":
        """Factory method to instantiate and load a BM25Retriever from JSON."""
        retriever = cls(tokenizer=tokenizer)
        retriever.load_from_json(filepath)
        return retriever
