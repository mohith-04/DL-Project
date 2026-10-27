"""
Extract and clean text from PDF files and plain-text files.
Uses PyMuPDF (fitz) for PDF parsing.
"""
import io
import re
from typing import Optional


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> Optional[str]:
    """
    Extract text from a PDF file given its bytes.
    Returns cleaned text or None on failure.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n\n".join(pages)
        if text and len(text.strip()) > 100:
            return _clean_text(text)
        return None
    except Exception as e:
        print(f"[pdf_parser] PyMuPDF failed: {e}")
        return None


def extract_text_from_file(filename: str, file_bytes: bytes) -> Optional[str]:
    """
    Dispatch to the correct extractor based on file extension.
    Supports: .pdf, .txt, .md
    Returns cleaned text or None on failure.
    """
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif fname.endswith((".txt", ".md")):
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            if text and len(text.strip()) > 50:
                return _clean_text(text)
        except Exception as e:
            print(f"[pdf_parser] Text decode failed: {e}")
        return None
    else:
        # Try UTF-8 decode as generic text fallback
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            if text and len(text.strip()) > 50:
                return _clean_text(text)
        except Exception:
            pass
        return None
