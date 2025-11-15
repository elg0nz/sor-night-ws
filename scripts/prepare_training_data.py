#!/usr/bin/env python3
"""
Prepare High-Quality Training Data

This script re-processes the corpus to create higher-quality training examples:
1. Filters out titles, headings, and metadata
2. Combines related chunks for better context
3. Ensures minimum quality thresholds
4. Splits into train/eval sets

Run this before train_openai.py to get better training data.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"
TRAIN_OUTPUT = DATA_DIR / "train.jsonl"
EVAL_OUTPUT = DATA_DIR / "eval.jsonl"

# Quality filters
MIN_WORDS = 40
MAX_WORDS = 500
MIN_CHARS = 200
EXCLUDE_TYPES = {"heading", "title"}
EVAL_SPLIT = 0.15  # 15% for evaluation

# Chunk combination settings
COMBINE_SEQUENTIAL = True  # Combine sequential chunks from same work
MAX_COMBINED_WORDS = 400  # Max words when combining chunks


def should_keep_chunk(text: str, metadata: Dict[str, Any]) -> bool:
    """
    Determine if a chunk meets quality criteria.

    Args:
        text: The text content
        metadata: Chunk metadata

    Returns:
        True if chunk should be kept
    """
    # Filter by chunk type
    chunk_type = metadata.get("chunk_type", "").lower()
    if chunk_type in EXCLUDE_TYPES:
        return False

    # Filter by length
    if len(text) < MIN_CHARS:
        return False

    # Filter by word count
    word_count = metadata.get("word_count", len(text.split()))
    if word_count < MIN_WORDS or word_count > MAX_WORDS:
        return False

    # Filter out metadata-like content
    if text.startswith("Title :") or text.startswith("Credits :"):
        return False

    if text.startswith("Original publication :"):
        return False

    # Must have some substance (not just numbers or very short phrases)
    if word_count < 10 and not any(char.isalpha() for char in text[:50]):
        return False

    return True


def combine_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine sequential chunks from the same work for better context.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        List of potentially combined chunks
    """
    if not COMBINE_SEQUENTIAL:
        return chunks

    # Group by parent document
    by_parent = defaultdict(list)
    for chunk in chunks:
        parent = chunk["metadata"].get("parent_document", "")
        if parent:
            by_parent[parent].append(chunk)

    combined = []

    for parent, parent_chunks in by_parent.items():
        # Sort by chunk number
        parent_chunks.sort(key=lambda x: x["metadata"].get("chunk_number", 0))

        i = 0
        while i < len(parent_chunks):
            current = parent_chunks[i]
            current_text = current["text"]
            current_words = current["metadata"].get("word_count", len(current_text.split()))

            # Try to combine with next chunks if they're sequential and related
            j = i + 1
            combined_text = current_text
            combined_words = current_words

            while j < len(parent_chunks) and combined_words < MAX_COMBINED_WORDS:
                next_chunk = parent_chunks[j]
                next_words = next_chunk["metadata"].get("word_count", len(next_chunk["text"].split()))

                # Check if chunks are sequential
                current_num = current["metadata"].get("chunk_number", 0)
                next_num = next_chunk["metadata"].get("chunk_number", 0)

                if next_num == current_num + (j - i):
                    # Same type and sequential - combine
                    if combined_words + next_words <= MAX_COMBINED_WORDS:
                        combined_text += "\n\n" + next_chunk["text"]
                        combined_words += next_words
                        j += 1
                    else:
                        break
                else:
                    break

            # Create combined chunk
            new_chunk = {
                "text": combined_text,
                "metadata": current["metadata"].copy()
            }
            new_chunk["metadata"]["word_count"] = combined_words
            if j > i + 1:
                new_chunk["metadata"]["combined_chunks"] = j - i

            combined.append(new_chunk)
            i = j

    # Add chunks without parent (shouldn't happen, but just in case)
    for chunk in chunks:
        if not chunk["metadata"].get("parent_document"):
            combined.append(chunk)

    return combined


def load_and_filter_corpus() -> List[Dict[str, Any]]:
    """
    Load corpus and apply quality filters.

    Returns:
        List of high-quality chunks
    """
    print(f"Loading corpus from {CORPUS_FILE}...")

    chunks = []
    total = 0
    filtered = {"too_short": 0, "bad_type": 0, "too_long": 0, "metadata": 0}

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            item = json.loads(line)
            text = item.get("text", "").strip()
            metadata = item.get("metadata", {})

            # Apply filters
            chunk_type = metadata.get("chunk_type", "").lower()
            word_count = metadata.get("word_count", len(text.split()))

            if chunk_type in EXCLUDE_TYPES:
                filtered["bad_type"] += 1
                continue

            if len(text) < MIN_CHARS:
                filtered["too_short"] += 1
                continue

            if word_count < MIN_WORDS:
                filtered["too_short"] += 1
                continue

            if word_count > MAX_WORDS:
                filtered["too_long"] += 1
                continue

            if text.startswith(("Title :", "Credits :", "Original publication :")):
                filtered["metadata"] += 1
                continue

            chunks.append({"text": text, "metadata": metadata})

    print(f"✓ Loaded {total} chunks")
    print(f"✓ Kept {len(chunks)} high-quality chunks")
    print(f"  Filtered out:")
    print(f"    - Bad type (heading/title): {filtered['bad_type']}")
    print(f"    - Too short: {filtered['too_short']}")
    print(f"    - Too long: {filtered['too_long']}")
    print(f"    - Metadata: {filtered['metadata']}")

    return chunks


def split_train_eval(chunks: List[Dict[str, Any]]) -> tuple:
    """
    Split chunks into training and evaluation sets.

    Args:
        chunks: List of chunks

    Returns:
        Tuple of (train_chunks, eval_chunks)
    """
    # Shuffle for random split
    random.seed(42)  # Reproducible split
    shuffled = chunks.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * (1 - EVAL_SPLIT))
    train = shuffled[:split_idx]
    eval_set = shuffled[split_idx:]

    return train, eval_set


def save_dataset(chunks: List[Dict[str, Any]], output_file: Path):
    """
    Save chunks to JSONL file.

    Args:
        chunks: List of chunks
        output_file: Output file path
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(chunks)} examples to {output_file.name}")


def main():
    """Main execution."""
    print("=" * 60)
    print("Preparing High-Quality Training Data for Sor Juana Fine-tuning")
    print("=" * 60)
    print()

    # Step 1: Load and filter
    print("Step 1: Loading and filtering corpus...")
    chunks = load_and_filter_corpus()
    print()

    # Step 2: Combine related chunks
    if COMBINE_SEQUENTIAL:
        print("Step 2: Combining sequential chunks...")
        original_count = len(chunks)
        chunks = combine_chunks(chunks)
        print(f"✓ Combined into {len(chunks)} chunks (from {original_count})")
        print()

    # Step 3: Split into train/eval
    print("Step 3: Splitting into train/eval sets...")
    train, eval_set = split_train_eval(chunks)
    print(f"✓ Train: {len(train)} examples ({(1-EVAL_SPLIT)*100:.0f}%)")
    print(f"✓ Eval: {len(eval_set)} examples ({EVAL_SPLIT*100:.0f}%)")
    print()

    # Step 4: Save
    print("Step 4: Saving datasets...")
    save_dataset(train, TRAIN_OUTPUT)
    save_dataset(eval_set, EVAL_OUTPUT)
    print()

    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Training examples: {len(train)}")
    print(f"  Evaluation examples: {len(eval_set)}")
    print(f"  Total: {len(chunks)}")
    print()
    print("Next steps:")
    print("  1. Review the generated files:")
    print(f"     - {TRAIN_OUTPUT}")
    print(f"     - {EVAL_OUTPUT}")
    print("  2. Run: python scripts/train_openai.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
