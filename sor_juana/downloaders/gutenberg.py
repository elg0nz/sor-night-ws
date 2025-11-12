"""Project Gutenberg downloader."""

from typing import List, Dict, Any, Optional
import requests
from .base import BaseDownloader
from ..processing import extract_gutenberg_chunks, clean_text
from ..config import REQUEST_TIMEOUT


class GutenbergDownloader(BaseDownloader):
    """Downloads Sor Juana texts from Project Gutenberg."""

    HTML_CANDIDATE_URLS = [
        "https://www.gutenberg.org/cache/epub/74087/pg74087-images.html",
        "https://www.gutenberg.org/cache/epub/74087/pg74087-h.html",
        "https://www.gutenberg.org/cache/epub/74087/pg74087.html",
        "https://www.gutenberg.org/files/74087/74087-h/74087-h.htm",
    ]
    TEXT_FALLBACK_URL = "https://www.gutenberg.org/cache/epub/74087/pg74087.txt"

    @property
    def source_name(self) -> str:
        return "gutenberg"

    def _fetch_first_available(self, urls: List[str]) -> Optional[requests.Response]:
        for url in urls:
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    response.url = url  # type: ignore[attr-defined]
                    return response
            except requests.RequestException:
                continue
        return None

    def download(self) -> List[Dict[str, Any]]:
        """Download text from Project Gutenberg."""
        html_response = self._fetch_first_available(self.HTML_CANDIDATE_URLS)

        if html_response:
            try:
                chunks = extract_gutenberg_chunks(html_response.text)
                if chunks:
                    return self._chunks_to_items(chunks, html_response.url)
            except Exception:
                pass

        text_response = self._fetch_first_available([self.TEXT_FALLBACK_URL])
        if text_response:
            fallback_text = clean_text(text_response.text, self.source_name)
            metadata = {
                "source": self.source_name,
                "title": "Obras selectas",
                "url": text_response.url,
                "genre": "poetry_prose",
            }
            return [{"text": fallback_text, "metadata": metadata}]

        raise ValueError("Gutenberg download failed: no available HTML or text sources.")

    def _chunks_to_items(self, chunks: List[Dict[str, Any]], source_url: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for chunk in chunks:
            section_title = (
                chunk.get("section_title")
                or (chunk.get("section_hierarchy") or [None])[-1]
                or chunk.get("parent_title")
                or "Obras selectas"
            )

            metadata = {
                "source": self.source_name,
                "title": section_title,
                "url": source_url,
                "genre": "poetry_prose",
                "parent_document": chunk.get("parent_title"),
                "chunk_number": chunk.get("chunk_number"),
                "chunk_type": chunk.get("chunk_type"),
                "section_hierarchy": chunk.get("section_hierarchy"),
                "section_title": chunk.get("section_title"),
                "word_count": chunk.get("word_count"),
            }

            if chunk.get("page_number") is not None:
                metadata["page_number"] = chunk["page_number"]
            if chunk.get("anchor"):
                metadata["anchor"] = chunk["anchor"]

            results.append({"text": chunk["text"], "metadata": metadata})

        return results
