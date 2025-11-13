"""Wikisource downloader."""

from typing import List, Dict, Any
import mwclient
from .base import BaseDownloader
from ..processing import extract_wikisource_chunks, clean_text
from ..config import WIKISOURCE_PAGE_LIMIT


class WikisourceDownloader(BaseDownloader):
    """Downloads Sor Juana texts from Wikisource."""

    WIKISOURCE_SITE = "es.wikisource.org"
    CATEGORY = "Categoría:Sor Juana Inés de la Cruz"  # Spanish category name

    @property
    def source_name(self) -> str:
        return "wikisource"

    def download(self) -> List[Dict[str, Any]]:
        """Download texts from Wikisource using MediaWiki API."""
        try:
            # Connect to Spanish Wikisource
            site = mwclient.Site(self.WIKISOURCE_SITE, scheme="https")
            # Get pages from the category
            category = site.Pages[self.CATEGORY]
            pages = list(category)
            all_items = []

            for page in pages[:WIKISOURCE_PAGE_LIMIT]:
                try:
                    page_name = page.name
                    source_url = f"https://{self.WIKISOURCE_SITE}/wiki/{page_name}"

                    # Get page content - page.text() is a method that returns the wiki markup
                    content = page.text()

                    if not content or len(content.strip()) < 50:
                        continue

                    # Try semantic chunking first
                    try:
                        chunks = extract_wikisource_chunks(content, page_name, source_url)
                        if chunks:
                            all_items.extend(self._chunks_to_items(chunks, page_name, source_url))
                            continue
                    except Exception:
                        pass

                    # Fallback to simple text extraction
                    text = clean_text(content, self.source_name)

                    if len(text) > 100:  # Filter short pages
                        metadata = {
                            "source": self.source_name,
                            "title": page_name,
                            "url": source_url,
                        }
                        all_items.append({"text": text, "metadata": metadata})

                except Exception:
                    # Skip pages that fail to download
                    continue

            return all_items

        except Exception as e:
            raise ValueError(f"Wikisource download failed: {str(e)}")

    def _chunks_to_items(self, chunks: List[Dict[str, Any]], page_name: str, source_url: str) -> List[Dict[str, Any]]:
        """Convert chunk dictionaries to items for database storage."""
        items = []

        for chunk in chunks:
            metadata = {
                "source": self.source_name,
                "title": page_name,
                "url": source_url,
                "chunk_number": chunk.get("chunk_number"),
                "chunk_type": chunk.get("chunk_type"),
                "section_title": chunk.get("section_title"),
                "section_hierarchy": chunk.get("section_hierarchy", []),
            }
            items.append({"text": chunk["text"], "metadata": metadata})

        return items
