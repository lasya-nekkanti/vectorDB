# Document → Vector Database Pipeline

Converts a 10-20 page document (.txt, .pdf, or .docx) into a searchable
local vector database using ChromaDB and sentence-transformers.

## How it works

1. **Load** — `document_loader.py` extracts raw text from your file.
2. **Chunk** — `chunker.py` splits the text into overlapping sentence-based
   chunks (default: 5 sentences per chunk, 1 sentence overlap).
3. **Embed** — `vector_store.py` uses the `all-MiniLM-L6-v2` model from
   sentence-transformers to turn each chunk into a 384-dimensional vector.
   This model is free and runs fully locally (downloads once, then works
   offline).
4. **Store** — Each chunk's text, embedding, and metadata (source file,
   chunk index) are saved into a persistent ChromaDB collection on disk
   at `chroma_db/`.
5. **Query** — `query.py` embeds your search question the same way and
   asks ChromaDB for the nearest matching chunks.

## Setup

```bash
pip install -r requirements.txt
```

The first time you run ingestion, sentence-transformers will download the
`all-MiniLM-L6-v2` model (~90MB) from Hugging Face. This requires an
internet connection once; after that it's cached locally and works offline.

## Usage

### 1. Build the vector database

```bash
python ingest.py path/to/your_document.pdf
```

Use `--reset` to wipe the existing collection before ingesting (useful if
you're re-running on an updated document):

```bash
python ingest.py path/to/your_document.pdf --reset
```

A sample document is included for testing:

```bash
python ingest.py sample_data/sample_document.txt --reset
```

### 2. Query the vector database

Single query:
```bash
python query.py "What is chunking and why is it needed?"
```

Interactive mode (ask multiple questions in one session):
```bash
python query.py
```

## Project structure

```
doc2vectordb/
├── document_loader.py   # Extracts text from .txt / .pdf / .docx
├── chunker.py            # Splits text into overlapping sentence chunks
├── vector_store.py        # ChromaDB + sentence-transformers wrapper
├── ingest.py               # Main pipeline: load -> chunk -> embed -> store
├── query.py                # Search the vector database
├── requirements.txt
├── sample_data/
│   └── sample_document.txt
└── chroma_db/             # Created automatically — persistent vector storage
```

## Customizing chunk size

In `ingest.py`, adjust these parameters to `ingest_document()`:

- `max_sentences_per_chunk` (default 5) — larger chunks keep more context
  but reduce precision; smaller chunks are more precise but lose context.
- `overlap_sentences` (default 1) — how many trailing sentences are
  repeated into the next chunk, to avoid losing context at chunk boundaries.

## Talking about this project in interviews

A few things worth understanding, not just running:

- **Why embeddings instead of keyword search?** Embeddings capture
  semantic meaning, so a query like "how do I split documents" can match
  a chunk that says "chunking strategies" without sharing exact words.
- **Why chunk at all?** Embedding models have input length limits, and
  smaller, focused chunks produce more precise, less "diluted" vectors
  than embedding an entire document at once.
- **Why overlap chunks?** Without overlap, an idea that spans a chunk
  boundary can lose context — half the explanation lands in chunk N,
  half in chunk N+1, and neither chunk fully represents the idea alone.
- **Distance metric** — ChromaDB defaults to cosine similarity for text
  embeddings; lower distance = more semantically similar.
- **Persistence** — `PersistentClient` writes to disk, so you embed once
  and query many times without recomputing vectors.
