"""Text processing utilities for cleaning and deduplication."""

import re
from typing import List, Set
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from datasketch import MinHash, MinHashLSH
from .config import MINHASH_THRESHOLD, MINHASH_NUM_PERM

# NLTK setup
nltk.download("punkt", quiet=True)


def clean_text(raw_text: str, source: str) -> str:
    """
    Clean and normalize text from various sources.

    Args:
        raw_text: Raw text content
        source: Source identifier (gutenberg, wikisource, bvmc)

    Returns:
        Cleaned and normalized text
    """
    # Remove Gutenberg headers/license
    if source == "gutenberg":
        raw_text = re.sub(r"^\*\*\*.*?\*\*\*\n", "", raw_text, flags=re.MULTILINE | re.DOTALL)
        raw_text = re.sub(r"END OF THE PROJECT GUTENBERG.*", "", raw_text, flags=re.MULTILINE)

    # Strip HTML/XML tags
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text()

    # Normalize whitespace, remove short lines
    text = re.sub(r"\s+", " ", text).strip()
    text = "\n".join(line for line in text.split("\n") if len(line) > 20)

    # Basic tokenization for sentences (Spanish-friendly)
    sentences = sent_tokenize(text, language="spanish")
    return " ".join(sentences)


class TextDeduplicator:
    """Deduplicates texts using MinHash LSH."""

    def __init__(self, threshold: float = MINHASH_THRESHOLD, num_perm: int = MINHASH_NUM_PERM):
        """
        Initialize deduplicator.

        Args:
            threshold: Jaccard similarity threshold for duplicates
            num_perm: Number of permutations for MinHash
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}

    def _compute_minhash(self, text: str) -> MinHash:
        """Compute MinHash for a text."""
        m = MinHash(num_perm=self.num_perm)
        sentences = sent_tokenize(text, language="spanish")
        for sentence in sentences:
            for word in sentence.split():
                m.update(word.encode("utf8"))
        return m

    def add_text(self, text_id: int, text: str) -> None:
        """Add a text to the deduplication index."""
        minhash = self._compute_minhash(text)
        self.lsh.insert(text_id, minhash)
        self.minhashes[text_id] = minhash

    def find_duplicates(self) -> Set[int]:
        """
        Find duplicate text IDs.

        Returns:
            Set of text IDs that are duplicates
        """
        duplicates = set()

        for text_id, minhash in self.minhashes.items():
            candidates = self.lsh.query(minhash)

            for candidate_id in candidates:
                if candidate_id != text_id and candidate_id in self.minhashes:
                    similarity = minhash.jaccard(self.minhashes[candidate_id])
                    if similarity > self.threshold:
                        # Mark the higher ID as duplicate (keep lower ID)
                        duplicates.add(max(text_id, candidate_id))

        return duplicates

    def deduplicate_texts(self, texts: List[dict]) -> List[int]:
        """
        Find duplicate IDs from a list of texts.

        Args:
            texts: List of dicts with 'id' and 'text' keys

        Returns:
            List of duplicate text IDs to remove
        """
        for item in texts:
            self.add_text(item["id"], item["text"])

        return list(self.find_duplicates())
