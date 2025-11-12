"""Base downloader class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDownloader(ABC):
    """Abstract base class for text downloaders."""

    @abstractmethod
    def download(self) -> List[Dict[str, Any]]:
        """
        Download texts from source.

        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this source."""
        pass
