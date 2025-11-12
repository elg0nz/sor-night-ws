"""Wikisource downloader."""

from typing import List, Dict, Any
import mwclient
from .base import BaseDownloader
from ..processing import clean_text
from ..config import WIKISOURCE_PAGE_LIMIT


class WikisourceDownloader(BaseDownloader):
    """Downloads Sor Juana texts from Wikisource."""

    WIKISOURCE_SITE = "es.wikisource.org"
    CATEGORY = "Category:Sor Juana Inés de la Cruz"

    @property
    def source_name(self) -> str:
        return "wikisource"

    def download(self) -> List[Dict[str, Any]]:
        """Download texts from Wikisource."""
        try:
            site = mwclient.Site(self.WIKISOURCE_SITE)
            pages = list(site.Pages[self.CATEGORY])
            texts = []

            for page in pages[:WIKISOURCE_PAGE_LIMIT]:
                try:
                    content = page.edit()
                    text = clean_text(content, self.source_name)

                    if len(text) > 100:  # Filter short pages
                        metadata = {
                            "source": self.source_name,
                            "title": page.name,
                            "url": f"https://{self.WIKISOURCE_SITE}/wiki/{page.name}",
                        }
                        texts.append({"text": text, "metadata": metadata})

                except Exception:
                    continue

            return texts

        except Exception as e:
            raise ValueError(f"Wikisource download failed: {str(e)}")
