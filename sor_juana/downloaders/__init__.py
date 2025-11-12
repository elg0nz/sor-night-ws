"""Text downloaders for various sources."""

from .base import BaseDownloader
from .gutenberg import GutenbergDownloader
from .wikisource import WikisourceDownloader
from .bvmc import BVMCDownloader

__all__ = [
    "BaseDownloader",
    "GutenbergDownloader",
    "WikisourceDownloader",
    "BVMCDownloader",
]
