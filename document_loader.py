"""
document_loader.py
-------------------
Reads raw text out of a source document.
Supports .txt, .pdf, and .docx files.

Each loader function takes a file path and returns a single string
containing the full extracted text of the document.
"""

import os


def load_txt(file_path: str) -> str:
    """Read a plain text file and return its contents."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file, page by page, and join it together."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def load_docx(file_path: str) -> str:
    """Extract text from a Word (.docx) file, paragraph by paragraph."""
    import docx

    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_document(file_path: str) -> str:
    """
    Dispatch to the correct loader based on file extension.
    Returns the full raw text of the document as a single string.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types are .txt, .pdf, .docx"
        )


if __name__ == "__main__":
    # Quick manual test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path_to_document>")
        sys.exit(1)

    text = load_document(sys.argv[1])
    print(f"Extracted {len(text)} characters.")
    print("---- Preview ----")
    print(text[:500])
