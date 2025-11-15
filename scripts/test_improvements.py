#!/usr/bin/env python3
"""
Test the improved data quality filters without uploading to OpenAI.

This shows how many examples will be kept vs filtered out.
"""

import json
import random
from pathlib import Path

# Configuration (copied from train_openai.py)
MIN_WORD_COUNT = 40
MAX_WORD_COUNT = 500
MIN_CHAR_LENGTH = 200
EXCLUDE_CHUNK_TYPES = {"heading", "title"}


def get_diverse_prompt(text: str, metadata: dict) -> str:
    """Generate diverse prompts (simplified version)."""
    genre = metadata.get("genre", "").lower()

    poetry_prompts = [
        "Escribe un soneto barroco sobre el amor imposible y el desengaño.",
        "Compón versos filosóficos sobre la naturaleza del conocimiento y la sabiduría.",
        "Crea una redondilla ingeniosa criticando la hipocresía de los hombres.",
    ]

    prose_prompts = [
        "Escribe una reflexión filosófica en prosa sobre la educación de las mujeres.",
        "Redacta una carta defendiendo el derecho de la mujer al estudio.",
    ]

    general_prompts = [
        "Escribe en tu característico estilo barroco, con cultismos y referencias mitológicas.",
        "Compón un texto literario con conceptos agudos y lenguaje elevado.",
    ]

    if "poes" in genre or "poetry" in genre:
        return random.choice(poetry_prompts)
    elif "letter" in genre or "carta" in genre or "prose" in genre:
        return random.choice(prose_prompts)
    else:
        return random.choice(general_prompts)

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"


def test_filters(input_file: Path):
    """Test how many examples pass the quality filters."""
    print(f"\n{'='*60}")
    print(f"Testing filters on: {input_file.name}")
    print(f"{'='*60}\n")

    total = 0
    kept = 0
    skip_reasons = {
        "too_short_chars": 0,
        "too_short_words": 0,
        "too_long_words": 0,
        "bad_type": 0,
    }

    sample_prompts = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            item = json.loads(line)
            text = item.get("text", "").strip()
            metadata = item.get("metadata", {})

            # Apply same filters as train_openai.py
            word_count = metadata.get("word_count", len(text.split()))
            chunk_type = metadata.get("chunk_type", "").lower()

            # Filter 1: chunk type
            if chunk_type in EXCLUDE_CHUNK_TYPES:
                skip_reasons["bad_type"] += 1
                continue

            # Filter 2: char length
            if len(text) < MIN_CHAR_LENGTH:
                skip_reasons["too_short_chars"] += 1
                continue

            # Filter 3: word count
            if word_count < MIN_WORD_COUNT:
                skip_reasons["too_short_words"] += 1
                continue

            if word_count > MAX_WORD_COUNT:
                skip_reasons["too_long_words"] += 1
                continue

            # This example passes all filters
            kept += 1

            # Collect sample prompts
            if len(sample_prompts) < 10:
                prompt = get_diverse_prompt(text, metadata)
                sample_prompts.append({
                    "prompt": prompt,
                    "word_count": word_count,
                    "genre": metadata.get("genre", "unknown"),
                })

    # Print results
    print(f"Total examples: {total}")
    print(f"Kept: {kept} ({kept/total*100:.1f}%)")
    print(f"Filtered: {total - kept} ({(total-kept)/total*100:.1f}%)")
    print(f"\nFilter breakdown:")
    print(f"  - Bad type (heading/title): {skip_reasons['bad_type']}")
    print(f"  - Too short (<{MIN_CHAR_LENGTH} chars): {skip_reasons['too_short_chars']}")
    print(f"  - Too few words (<{MIN_WORD_COUNT}): {skip_reasons['too_short_words']}")
    print(f"  - Too many words (>{MAX_WORD_COUNT}): {skip_reasons['too_long_words']}")

    print(f"\n{'='*60}")
    print("Sample Prompts (showing prompt diversity):")
    print(f"{'='*60}\n")
    for i, sample in enumerate(sample_prompts, 1):
        print(f"{i}. [{sample['genre']}] ({sample['word_count']} words)")
        print(f"   {sample['prompt']}")
        print()

    return kept, total


def main():
    """Test both train and eval files."""
    print("\n" + "="*60)
    print("Testing Improved Data Quality Filters")
    print("="*60)

    print(f"\nFilter settings:")
    print(f"  - Min characters: {MIN_CHAR_LENGTH}")
    print(f"  - Min words: {MIN_WORD_COUNT}")
    print(f"  - Max words: {MAX_WORD_COUNT}")
    print(f"  - Excluded types: {EXCLUDE_CHUNK_TYPES}")

    # Test train file
    train_kept, train_total = test_filters(TRAIN_FILE)

    # Test eval file
    eval_kept, eval_total = test_filters(EVAL_FILE)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    print(f"Training set: {train_kept}/{train_total} examples ({train_kept/train_total*100:.1f}%)")
    print(f"Eval set: {eval_kept}/{eval_total} examples ({eval_kept/eval_total*100:.1f}%)")
    print(f"Total: {train_kept + eval_kept}/{train_total + eval_total} examples")
    print()

    if train_kept < 100:
        print("⚠️  WARNING: Less than 100 training examples!")
        print("   Consider:")
        print("   1. Lowering MIN_WORD_COUNT (currently {})".format(MIN_WORD_COUNT))
        print("   2. Lowering MIN_CHAR_LENGTH (currently {})".format(MIN_CHAR_LENGTH))
        print("   3. Increasing MAX_WORD_COUNT (currently {})".format(MAX_WORD_COUNT))
        print("   4. Building a larger corpus from additional sources")
    else:
        print("✓ Dataset size looks good for fine-tuning!")

    print()
    print("Next steps:")
    print("  - If satisfied, run: python scripts/train_openai.py")
    print("  - To adjust filters, edit MIN_WORD_COUNT, MAX_WORD_COUNT,")
    print("    MIN_CHAR_LENGTH in scripts/train_openai.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
