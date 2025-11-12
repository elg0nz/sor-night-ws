"""Text processing utilities for cleaning, chunking, and deduplication."""

import re
from typing import Any, Dict, List, Optional, Set
from bs4 import BeautifulSoup, NavigableString, Tag
import nltk
from nltk.tokenize import sent_tokenize
from datasketch import MinHash, MinHashLSH
from .config import MINHASH_THRESHOLD, MINHASH_NUM_PERM


def _ensure_punkt_resources() -> None:
    """Ensure NLTK Punkt resources required for Spanish tokenization are present."""
    resources = (
        ("punkt", "tokenizers/punkt/spanish.pickle"),
        ("punkt_tab", "tokenizers/punkt_tab/spanish/"),
    )

    for package, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            downloaded = nltk.download(package, quiet=True)
            if not downloaded:
                raise LookupError(f"Failed to download required NLTK resource '{package}'.")
            # Re-check to make sure the resource is now available.
            nltk.data.find(path)


_ensure_punkt_resources()


PAGE_PATTERN = re.compile(r"\[Pg\s*(\d+)\]", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_whitespace(text: str, preserve_newlines: bool = False) -> str:
    if preserve_newlines:
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _contains_gutenberg_license(text: str) -> bool:
    upper = text.upper()
    return "PROJECT GUTENBERG" in upper and ("LICENSE" in upper or "*** START OF" in upper or "*** END OF" in upper)


def clean_text(raw_text: str, source: str) -> str:
    """
    Clean and normalize plain text from sources that do not require semantic chunking.

    Args:
        raw_text: Raw text content.
        source: Source identifier (wikisource, bvmc, etc.).

    Returns:
        Cleaned and normalized text.
    """
    text = raw_text
    if source == "gutenberg":
        # Legacy compatibility: fallback when only plain text is available.
        text = re.sub(r"^\*\*\*.*?\*\*\*\n", "", text, flags=re.MULTILINE | re.DOTALL)
        text = re.sub(r"END OF THE PROJECT GUTENBERG.*", "", text, flags=re.MULTILINE)

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    text = "\n".join(line for line in text.split("\n") if len(line) > 20)
    sentences = sent_tokenize(text, language="spanish")
    return " ".join(sentences)


def extract_gutenberg_chunks(raw_html: str) -> List[Dict[str, Any]]:
    """
    Parse Project Gutenberg HTML content into semantically meaningful chunks.

    Args:
        raw_html: HTML document string from Project Gutenberg.

    Returns:
        List of chunk dictionaries containing text and rich metadata.
    """

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    body = soup.find("body")
    if not body:
        return []

    document_title = _normalize_whitespace(soup.title.get_text()) if soup.title else "Gutenberg Document"
    chunks: List[Dict[str, Any]] = []
    heading_stack: Dict[int, str] = {}
    current_anchor: Optional[str] = None
    last_page_number: Optional[int] = None

    def current_section_title() -> Optional[str]:
        if not heading_stack:
            return None
        max_level = max(heading_stack.keys())
        return heading_stack[max_level]

    def section_hierarchy() -> List[str]:
        return [heading_stack[key] for key in sorted(heading_stack.keys())]

    def add_chunk(text: str, chunk_type: str, *, preserve_newlines: bool = False) -> None:
        nonlocal last_page_number
        if not text:
            return

        if _contains_gutenberg_license(text):
            return

        normalized = _normalize_whitespace(text, preserve_newlines=preserve_newlines)
        if not normalized:
            return

        if chunk_type not in {"heading", "poetry"} and len(normalized) < 40:
            return

        chunk: Dict[str, Any] = {
            "text": normalized if chunk_type != "poetry" else text.strip(),
            "chunk_number": len(chunks) + 1,
            "chunk_type": chunk_type,
            "parent_title": document_title,
            "section_title": current_section_title(),
            "section_hierarchy": section_hierarchy(),
        }

        if current_anchor:
            chunk["anchor"] = current_anchor

        if last_page_number is not None:
            chunk["page_number"] = last_page_number

        word_count = len(re.findall(r"\w+", chunk["text"], flags=re.UNICODE))
        chunk["word_count"] = word_count

        chunks.append(chunk)

    def extract_text(tag: Tag, *, preserve_newlines: bool = False) -> str:
        nonlocal last_page_number
        raw = tag.get_text("\n" if preserve_newlines else " ", strip=True)
        for match in PAGE_PATTERN.finditer(raw):
            last_page_number = int(match.group(1))
        cleaned = PAGE_PATTERN.sub("", raw)
        return cleaned

    def process_container(container: Tag) -> None:
        nonlocal current_anchor, last_page_number
        for child in container.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue

            if child.name == "span" and "pagenum" in (child.get("class") or []):
                text = child.get_text(strip=True)
                match = PAGE_PATTERN.search(text)
                if match:
                    last_page_number = int(match.group(1))
                continue

            if child.name in {"h1", "h2", "h3", "h4"}:
                level = int(child.name[1])
                heading_text = extract_text(child)
                if not heading_text:
                    continue

                for key in list(heading_stack.keys()):
                    if key >= level:
                        heading_stack.pop(key)

                heading_stack[level] = heading_text
                current_id = child.get("id") or child.get("name")
                if current_id:
                    current_anchor = current_id
                add_chunk(heading_text, "heading")
                continue

            if child.name == "p":
                paragraph_text = extract_text(child)
                add_chunk(paragraph_text, "paragraph")
                continue

            if child.name in {"blockquote", "q"}:
                quote_text = extract_text(child)
                add_chunk(quote_text, "blockquote")
                continue

            if child.name == "pre":
                pre_text = extract_text(child, preserve_newlines=True)
                add_chunk(pre_text, "preformatted", preserve_newlines=True)
                continue

            if child.name in {"ul", "ol"}:
                list_items = [extract_text(li) for li in child.find_all("li", recursive=False)]
                list_text = "\n".join(f"- {item}" for item in list_items if item)
                add_chunk(list_text, "list", preserve_newlines=True)
                continue

            if child.name == "div":
                classes = child.get("class") or []
                if any(cls in {"poem", "poetry", "verse"} for cls in classes):
                    poem_text = extract_text(child, preserve_newlines=True)
                    add_chunk(poem_text, "poetry", preserve_newlines=True)
                    continue
                if "chapter" in classes:
                    process_container(child)
                    continue

            process_container(child)

    process_container(body)
    return chunks


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
