"""
vector_store.py
----------------
Wraps ChromaDB + sentence-transformers into a simple interface for:
  1. Embedding text chunks
  2. Storing them in a persistent local vector database
  3. Querying the database with natural language

Embedding model: all-MiniLM-L6-v2
  - Free, runs locally (no API key, no internet needed after first download)
  - Produces 384-dimensional vectors
  - Good balance of speed and semantic quality for general text

ChromaDB persistence:
  - Stored on disk at `chroma_db/` so the database survives between runs
  - You build it once, then query it any time without re-embedding
"""

import os
import uuid

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        # Persistent client writes to disk so data survives between runs
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

        print(f"Loading embedding model '{embedding_model_name}'...")
        self.model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded.")

    def embed(self, texts: list[str]):
        """Convert a list of text chunks into embedding vectors."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def add_chunks(self, chunks: list[str], source_name: str):
        """
        Embed and store a list of text chunks in the vector database.

        Args:
            chunks: list of text chunks to store.
            source_name: name of the originating document, stored as
                metadata so you know where each chunk came from.
        """
        if not chunks:
            print("No chunks to add.")
            return

        embeddings = self.embed(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"source": source_name, "chunk_index": i} for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        print(f"Added {len(chunks)} chunks from '{source_name}' to the vector store.")

    def query(self, query_text: str, top_k: int = 3):
        """
        Embed a query string and return the most similar stored chunks.

        Returns a list of dicts: {text, source, chunk_index, distance}
        Lower distance = more similar.
        """
        query_embedding = self.embed([query_text])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        output = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            output.append(
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "chunk_index": meta.get("chunk_index"),
                    "distance": dist,
                }
            )
        return output

    def count(self) -> int:
        """Return how many chunks are currently stored."""
        return self.collection.count()

    def reset(self):
        """Delete all stored chunks (useful when re-ingesting from scratch)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        print("Vector store collection reset.")
