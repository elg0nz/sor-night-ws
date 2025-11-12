"""Project Gutenberg downloader."""

from typing import List, Dict, Any
import requests
from .base import BaseDownloader
from ..processing import clean_text
from ..config import REQUEST_TIMEOUT


class GutenbergDownloader(BaseDownloader):
    """Downloads Sor Juana texts from Project Gutenberg."""

    GUTENBERG_URL = "https://www.gutenberg.org/ebooks/74087.txt.utf-8"

    @property
    def source_name(self) -> str:
        return "gutenberg"

    def download(self) -> List[Dict[str, Any]]:
        """Download text from Project Gutenberg."""
        try:
            response = requests.get(self.GUTENBERG_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            text = clean_text(response.text, self.source_name)

            metadata = {
                "source": self.source_name,
                "title": "Obras selectas",
                "url": self.GUTENBERG_URL,
                "genre": "poetry_prose",
            }

            return [{"text": text, "metadata": metadata}]

        except Exception as e:
            raise ValueError(f"Gutenberg download failed: {str(e)}")
