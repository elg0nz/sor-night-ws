"""Biblioteca Virtual Miguel de Cervantes downloader."""

from typing import List, Dict, Any
import time
import requests
from .base import BaseDownloader
from ..processing import extract_bvmc_chunks, clean_text
from ..config import REQUEST_TIMEOUT


class BVMCDownloader(BaseDownloader):
    """Downloads Sor Juana texts from BVMC (Biblioteca Virtual Miguel de Cervantes)."""

    # BVMC obra URLs - these are HTML viewer URLs for Sor Juana works
    WORKS_URLS = [
        "https://www.cervantesvirtual.com/portales/sor_juana_ines_de_la_cruz/obra-visor/inundacion-castalida--0/html/",
        "https://www.cervantesvirtual.com/portales/sor_juana_ines_de_la_cruz/obra-visor/respuesta-a-sor-filotea-de-la-cruz--0/html/",
        "https://www.cervantesvirtual.com/portales/sor_juana_ines_de_la_cruz/obra-visor/parayso-sic-occidental-plantado-y-cultivado-por-la-liberal-benefica-mano-de-los-muy-catholicos-y-p-0/html/",
    ]

    # Headers to identify our bot and be respectful
    HEADERS = {
        "User-Agent": "SorJuanaCorpusBot/1.0 (Academic Research; +https://github.com/yourusername/sor-juana-corpus)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es,en;q=0.9",
    }

    @property
    def source_name(self) -> str:
        return "bvmc"

    def download(self) -> List[Dict[str, Any]]:
        """Download texts from BVMC with respectful rate limiting."""
        all_items = []

        for url in self.WORKS_URLS:
            try:
                # Add a small delay between requests to be respectful
                if len(all_items) > 0:
                    time.sleep(1)

                response = requests.get(url, headers=self.HEADERS, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

                # Try semantic chunking first
                try:
                    chunks = extract_bvmc_chunks(response.text, url)
                    if chunks:
                        all_items.extend(self._chunks_to_items(chunks, url))
                        continue
                except Exception:
                    pass

                # Fallback to simple text extraction
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.content, "html.parser")
                # Try multiple content selectors (BVMC uses different structures)
                content_div = (
                    soup.find("div", {"id": "contenido"})
                    or soup.find("main")
                    or soup.find("article")
                    or soup.find("div", class_="obra")
                    or soup.find("div", class_="content")
                    or soup.find("body")
                )
                text = clean_text(content_div.get_text() if content_div else response.text, self.source_name)

                if len(text) > 100:
                    title = url.split("--")[-1].split(".")[0]
                    metadata = {
                        "source": self.source_name,
                        "title": title,
                        "url": url,
                        "genre": "poetry_drama",
                    }
                    all_items.append({"text": text, "metadata": metadata})

            except Exception:
                # Log error but continue with other URLs
                continue

        return all_items

    def _chunks_to_items(self, chunks: List[Dict[str, Any]], source_url: str) -> List[Dict[str, Any]]:
        """Convert chunk dictionaries to items for database storage."""
        items = []
        title = chunks[0].get("parent_title", "BVMC Document") if chunks else "BVMC Document"

        for chunk in chunks:
            metadata = {
                "source": self.source_name,
                "title": title,
                "url": source_url,
                "genre": "poetry_drama",
                "chunk_number": chunk.get("chunk_number"),
                "chunk_type": chunk.get("chunk_type"),
                "section_title": chunk.get("section_title"),
                "section_hierarchy": chunk.get("section_hierarchy", []),
            }
            items.append({"text": chunk["text"], "metadata": metadata})

        return items
