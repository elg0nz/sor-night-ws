"""Biblioteca Virtual Miguel de Cervantes downloader."""

from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from .base import BaseDownloader
from ..processing import clean_text
from ..config import REQUEST_TIMEOUT


class BVMCDownloader(BaseDownloader):
    """Downloads Sor Juana texts from BVMC."""

    WORKS_URLS = [
        "https://www.cervantesvirtual.com/obra-visor/obra-inundacion-castalida--0/html/ff2a7d3c-82b1-11df-acc7-002185ce6064_2.html",
        "https://www.cervantesvirtual.com/obra-visor/obra-respuesta-a-sor-filotea-de-la-cruz--0/html/ff2a7d3c-82b1-11df-acc7-002185ce6064_4.html",
    ]

    @property
    def source_name(self) -> str:
        return "bvmc"

    def download(self) -> List[Dict[str, Any]]:
        """Download texts from BVMC."""
        texts = []

        for url in self.WORKS_URLS:
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")
                content_div = soup.find("div", {"id": "contenido"}) or soup.find("body")
                text = clean_text(content_div.get_text() if content_div else response.text, self.source_name)

                if len(text) > 100:
                    title = url.split("--")[-1].split(".")[0]
                    metadata = {"source": self.source_name, "title": title, "url": url, "genre": "poetry_drama"}
                    texts.append({"text": text, "metadata": metadata})

            except Exception:
                continue

        return texts
