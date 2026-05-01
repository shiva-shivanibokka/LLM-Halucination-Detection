"""
ingestor/ingestor.py

Handles extraction and chunking of source documents from:
  - Plain text strings
  - PDF file bytes
  - Web URLs (requests + BeautifulSoup, Playwright fallback for JS-heavy pages)

All sources are split into overlapping sentence-level chunks before embedding.
"""

import re
from typing import Optional

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def _split_into_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def _clean(text: str) -> str:
    """Collapse whitespace and remove control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_chunks(text: str) -> list[str]:
    """Chunk plain text directly."""
    cleaned = _clean(text)
    return _split_into_chunks(cleaned)


def extract_pdf_chunks(pdf_bytes: bytes) -> list[str]:
    """Extract text from PDF bytes and chunk it."""
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    full_text = "\n".join(pages)
    cleaned = _clean(full_text)
    return _split_into_chunks(cleaned)


def extract_url_chunks(url: str) -> list[str]:
    """Scrape a URL and chunk its text content. Falls back to Playwright for JS pages."""
    text = _fetch_with_requests(url)
    if not text or len(text.split()) < 50:
        text = _fetch_with_playwright(url)
    if not text:
        raise ValueError(f"Could not extract text from URL: {url}")
    cleaned = _clean(text)
    return _split_into_chunks(cleaned)


def _fetch_with_requests(url: str) -> Optional[str]:
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; HallucinationDetector/1.0)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator=" ")
    except Exception:
        return None


def _fetch_with_playwright(url: str) -> Optional[str]:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            content = page.inner_text("body")
            browser.close()
            return content
    except Exception:
        return None
