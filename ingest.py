"""
ingest.py
---------
Main entry point for building the vector database from a source document.

Usage:
    python ingest.py path/to/your/document.pdf
    python ingest.py path/to/your/document.docx
    python ingest.py path/to/your/document.txt

Pipeline:
    1. Load the document (document_loader.py)
    2. Split it into overlapping sentence-based chunks (chunker.py)
    3. Embed each chunk and store it in ChromaDB (vector_store.py)
"""

import os
import sys

from document_loader import load_document
from chunker import chunk_text
from vector_store import VectorStore


def ingest_document(
    file_path: str,
    max_sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
    reset_existing: bool = False,
):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    source_name = os.path.basename(file_path)

    print(f"\nStep 1/3: Loading document '{source_name}'...")
    raw_text = load_document(file_path)
    print(f" -> Extracted {len(raw_text)} characters.")

    print("\nStep 2/3: Splitting into chunks...")
    chunks = chunk_text(
        raw_text,
        max_sentences_per_chunk=max_sentences_per_chunk,
        overlap_sentences=overlap_sentences,
    )
    print(f" -> Created {len(chunks)} chunks "
          f"(~{max_sentences_per_chunk} sentences each, "
          f"{overlap_sentences} sentence overlap).")

    print("\nStep 3/3: Embedding and storing in ChromaDB...")
    store = VectorStore()
    if reset_existing:
        store.reset()
    store.add_chunks(chunks, source_name=source_name)

    total = store.count()
    print(f"\nDone. Vector database now contains {total} total chunks.")
    return store


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_document> [--reset]")
        sys.exit(1)

    doc_path = sys.argv[1]
    reset_flag = "--reset" in sys.argv

    ingest_document(doc_path, reset_existing=reset_flag)
