"""Dense vector retrieval using ChromaDB and configurable embedders.

Provides:
- In-memory or persistent ChromaDB collection management with cosine distance.
- Idempotent chunk indexing with metadata retention.
- Cosine similarity search with optional scheme pre-filtering.
- Source provenance preservation across all retrieved chunks.
"""

from pathlib import Path
from typing import Any

import chromadb
import chromadb.errors

from app.ingestion.models import TextChunk
from app.retrieval.embeddings import BaseEmbedder
from app.retrieval.models import SearchResult


class ChromaDenseRetriever:
    """Dense vector retriever backed by ChromaDB."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        persist_directory: Path | str | None = None,
        collection_name: str = "scheme_chunks",
    ) -> None:
        """Initialize the ChromaDenseRetriever.

        Args:
            embedder: Embedder instance implementing BaseEmbedder.
            persist_directory: Optional filesystem path for persistent storage.
                If None, an in-memory ephemeral client is used (ideal for testing).
            collection_name: Name of the ChromaDB collection.
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory else None

        if self.persist_directory is not None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client: chromadb.ClientAPI = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return the number of vectors stored in the collection."""
        return self._collection.count()

    def index_chunks(self, chunks: list[TextChunk]) -> int:
        """Index or update text chunks in the ChromaDB collection.

        Args:
            chunks: List of TextChunk objects to embed and store.

        Returns:
            Total count of indexed chunks in the collection.
        """
        if not chunks:
            return self._collection.count()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.metadata.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "doc_id": chunk.metadata.doc_id,
                    "filename": chunk.metadata.filename,
                    "page_number": chunk.metadata.page_number,
                    "chunk_index": chunk.metadata.chunk_index,
                    "char_count": chunk.metadata.char_count,
                    "scheme_name": chunk.metadata.scheme_name or "",
                }
            )

        embeddings = self.embedder.embed_texts(documents)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return self._collection.count()

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to doc_id from Chroma collection.

        Args:
            doc_id: Identifier of the document whose chunks should be removed.

        Returns:
            Remaining vector count in the collection.
        """
        if not doc_id:
            return self._collection.count()

        try:
            self._collection.delete(where={"doc_id": doc_id})
        except (chromadb.errors.ChromaError, RuntimeError, ValueError, KeyError):
            pass

        return self._collection.count()

    def search(
        self,
        query: str,
        top_k: int = 5,
        scheme_name: str | None = None,
    ) -> list[SearchResult]:
        """Perform dense vector search for a given query string.

        Args:
            query: User search query.
            top_k: Maximum number of ranked results to return.
            scheme_name: Optional government scheme name to filter by.

        Returns:
            List of SearchResult objects ordered by cosine similarity descending.
        """
        if not query.strip() or self._collection.count() == 0 or top_k <= 0:
            return []

        query_embedding = self.embedder.embed_query(query)

        where: dict[str, Any] | None = None
        if scheme_name:
            where = {"scheme_name": scheme_name}

        n_results = min(top_k, self._collection.count())

        try:
            query_output = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except (chromadb.errors.ChromaError, RuntimeError, ValueError, KeyError):
            return []

        ids = query_output.get("ids", [[]])[0]
        documents = query_output.get("documents", [[]])[0]
        metadatas = query_output.get("metadatas", [[]])[0]
        distances = query_output.get("distances", [[]])[0]

        if not ids:
            return []

        candidates: list[tuple[float, str, str, dict[str, Any]]] = []
        for chunk_id, doc_text, meta, dist in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            # In Chroma's cosine space: cosine_distance = 1.0 - cosine_similarity
            # Clamp similarity between -1.0 and 1.0
            similarity = max(-1.0, min(1.0, 1.0 - float(dist)))
            candidates.append((similarity, chunk_id, doc_text, meta))

        # Sort descending by similarity score; break ties deterministically by chunk_id
        candidates.sort(key=lambda item: (-item[0], item[1]))

        results: list[SearchResult] = []
        for rank, (score, chunk_id, doc_text, meta) in enumerate(candidates, start=1):
            scheme = meta.get("scheme_name") or None
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", ""),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc_text,
                    scheme_name=scheme,
                    score=score,
                    dense_score=score,
                    dense_rank=rank,
                )
            )

        return results
