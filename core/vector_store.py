"""
ingestor/vector_store.py

ChromaDB wrapper. Stores text chunks as embeddings using sentence-transformers.
One in-memory collection per process — ephemeral, resets on restart.
"""

import uuid
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "source_documents"
TOP_K = 5


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.Client()
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[str], source_label: str = "unknown") -> int:
        if not chunks:
            return 0
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source_label} for _ in chunks]
        self._collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def query(self, text: str, k: int = TOP_K) -> list[dict]:
        n = min(k, self._collection.count())
        if n == 0:
            return []
        results = self._collection.query(
            query_texts=[text],
            n_results=n,
            include=["documents", "distances"],
        )
        chunks = results["documents"][0]
        distances = results["distances"][0]
        return [
            {"chunk": chunk, "similarity": 1.0 - dist}
            for chunk, dist in zip(chunks, distances)
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
