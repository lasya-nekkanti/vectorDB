"""
chunker.py
----------
Splits a large block of text into smaller overlapping chunks.

Why chunk at all?
  Embedding models work best on small, focused pieces of text (a few
  sentences), not entire 10-20 page documents. Chunking also lets us
  return precise, relevant snippets when we search later, instead of
  dumping the whole document back at the user.

Why sentence-aware?
  Splitting purely by character count can cut a sentence in half,
  which damages the meaning the embedding model is trying to capture.
  Splitting by sentence keeps each chunk semantically whole.

Why overlap?
  If a sentence chunk falls right on a chunk boundary, important
  context can get separated from the next idea. A small overlap
  (a sentence or two carried over to the next chunk) keeps context
  intact across boundaries.
"""

import re


def split_into_sentences(text: str) -> list[str]:
    """
    Basic sentence splitter using punctuation boundaries.
    Not perfect (e.g. doesn't handle abbreviations like 'Dr.' specially),
    but works well for general prose without adding heavy NLP dependencies.
    """
    # Normalize whitespace/newlines first
    text = re.sub(r"\s+", " ", text).strip()

    # Split after ., !, or ? when followed by a space and a capital letter / end of string
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(]|$)", text)

    # Clean up empties
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    max_sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Group sentences into overlapping chunks.

    Args:
        text: full document text.
        max_sentences_per_chunk: how many sentences go into each chunk.
        overlap_sentences: how many trailing sentences from the previous
            chunk are repeated at the start of the next chunk.

    Returns:
        List of chunk strings.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    start = 0
    step = max(max_sentences_per_chunk - overlap_sentences, 1)

    while start < len(sentences):
        end = min(start + max_sentences_per_chunk, len(sentences))
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        if end == len(sentences):
            break
        start += step

    return chunks


if __name__ == "__main__":
    sample = (
        "Vector databases store embeddings for fast similarity search. "
        "They are commonly used in retrieval-augmented generation. "
        "ChromaDB is a popular open-source choice. "
        "It supports persistent local storage. "
        "Embeddings are generated using models like sentence-transformers. "
        "These models map text into high-dimensional vectors. "
        "Similar meanings end up close together in vector space."
    )

    result = chunk_text(sample, max_sentences_per_chunk=3, overlap_sentences=1)
    for i, c in enumerate(result):
        print(f"Chunk {i}: {c}\n")
