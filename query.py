"""
query.py
--------
Search the vector database you built with ingest.py.

Usage (single query):
    python query.py "your search question here"

Usage (interactive mode):
    python query.py
    (then type queries one at a time, 'exit' to quit)
"""

import sys

from vector_store import VectorStore


def print_results(results):
    if not results:
        print("No results found. Did you run ingest.py first?")
        return

    for rank, r in enumerate(results, start=1):
        print(f"\n#{rank}  (distance={r['distance']:.4f})  "
              f"source={r['source']}  chunk_index={r['chunk_index']}")
        print(f"    {r['text']}")


def run_single_query(query_text: str, top_k: int = 3):
    store = VectorStore()
    results = store.query(query_text, top_k=top_k)
    print(f"\nTop {top_k} results for: \"{query_text}\"")
    print_results(results)


def run_interactive(top_k: int = 3):
    store = VectorStore()
    print(f"\nVector database contains {store.count()} chunks.")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        query_text = input("Query> ").strip()
        if query_text.lower() in ("exit", "quit"):
            break
        if not query_text:
            continue

        results = store.query(query_text, top_k=top_k)
        print_results(results)
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        run_single_query(user_query)
    else:
        run_interactive()
