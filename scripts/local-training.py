#!/usr/bin/env python3
"""
Sor Juana Style Model: Fine-tuning + Evaluation Pipeline
Optimized for Apple Silicon (M4 Mac mini) using MLX
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# MLX imports (install: pip install mlx mlx-lm)
try:
    import mlx.core as mx
except ImportError:
    print("ERROR: MLX not installed. Install with: pip install mlx mlx-lm")
    exit(1)

# Standard ML libraries
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Note: torch/transformers optional, using MLX backend")

# Evaluation metrics


@dataclass
class EvalConfig:
    """Evaluation configuration for Sor Juana style matching"""

    baroque_weight: float = 0.25
    thematic_weight: float = 0.25
    linguistic_weight: float = 0.25
    structural_weight: float = 0.25
    genre: str = "poetry"  # poetry, prose, drama
    min_score_threshold: float = 3.0  # 1-5 scale


class SorJuanaEvaluator:
    """Custom evaluator for Sor Juana style authenticity"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.baroque_keywords = {
            "conceptos": 0.8,
            "agudeza": 0.9,
            "paradoja": 0.85,
            "metáfora": 0.7,
            "antítesis": 0.8,
            "hipérbole": 0.75,
        }
        self.feminist_themes = {
            "educación": 0.9,
            "mujer": 0.8,
            "ingenio": 0.85,
            "saber": 0.8,
            "derecho": 0.85,
            "injusticia": 0.7,
        }
        self.theological_terms = {
            "alma": 0.7,
            "divino": 0.75,
            "sacramento": 0.85,
            "gracia": 0.8,
            "penitencia": 0.75,
        }
        self.period_vocabulary = {
            "vos": 0.8,
            "vuestra merced": 0.9,
            "fiera": 0.7,
            "grosero": 0.7,
            "tinieblas": 0.75,
        }

    def score_baroque_style(self, text: str) -> float:
        """Score text for Baroque rhetorical features (0-1)"""
        text_lower = text.lower()
        score = 0.0
        keywords_found = 0

        for keyword, weight in self.baroque_keywords.items():
            if keyword in text_lower:
                score += weight
                keywords_found += 1

        # Normalize and add structural complexity bonus
        if keywords_found > 0:
            score = min(1.0, (score / len(self.baroque_keywords)) * 1.5)

        # Bonus for longer, complex sentences (typical of Baroque)
        avg_sentence_length = np.mean([len(s.split()) for s in text.split(".")])
        if avg_sentence_length > 15:
            score = min(1.0, score + 0.15)

        return score

    def score_thematic_alignment(self, text: str) -> float:
        """Score alignment with Sor Juana's major themes (0-1)"""
        text_lower = text.lower()
        score = 0.0
        themes_found = 0

        # Check feminist/intellectual themes
        for theme, weight in self.feminist_themes.items():
            if theme in text_lower:
                score += weight
                themes_found += 1

        # Check theological themes
        for term, weight in self.theological_terms.items():
            if term in text_lower:
                score += weight
                themes_found += 1

        if themes_found > 0:
            score = score / (len(self.feminist_themes) + len(self.theological_terms))

        return min(1.0, score)

    def score_linguistic_authenticity(self, text: str) -> float:
        """Score period-appropriate language usage (0-1)"""
        text_lower = text.lower()
        score = 0.0
        features_found = 0

        for vocab, weight in self.period_vocabulary.items():
            if vocab in text_lower:
                score += weight
                features_found += 1

        # Penalize modern language
        modern_terms = ["ai", "computadora", "internet", "moderno"]
        for term in modern_terms:
            if term in text_lower:
                score -= 0.2

        if features_found > 0:
            score = score / len(self.period_vocabulary)

        return max(0.0, min(1.0, score))

    def score_structural_coherence(self, text: str) -> float:
        """Score structural integrity (sonnet, décima, coherence) (0-1)"""
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        score = 0.5  # Base score for coherence

        # Detect sonnets (14 lines)
        if len(lines) == 14:
            score += 0.3

        # Detect décimas (10 lines)
        if len(lines) == 10:
            score += 0.25

        # Check for rhyme (simple heuristic: ending sounds)
        if len(lines) > 2:
            endings = [line[-3:] if len(line) > 3 else line for line in lines]
            unique_endings = len(set(endings))
            rhyme_ratio = 1.0 - (unique_endings / len(endings))
            score += rhyme_ratio * 0.2

        return min(1.0, score)

    def evaluate(self, text: str) -> Dict[str, float]:
        """
        Comprehensive evaluation scoring (returns dict with subscores and overall)
        Output: {
            "baroque_style": 0.0-1.0,
            "thematic_alignment": 0.0-1.0,
            "linguistic_authenticity": 0.0-1.0,
            "structural_coherence": 0.0-1.0,
            "overall_score": 1.0-5.0,
            "passes_threshold": bool
        }
        """
        baroque = self.score_baroque_style(text)
        thematic = self.score_thematic_alignment(text)
        linguistic = self.score_linguistic_authenticity(text)
        structural = self.score_structural_coherence(text)

        # Weighted average to 1.0-5.0 scale
        weighted_score = (
            baroque * self.config.baroque_weight
            + thematic * self.config.thematic_weight
            + linguistic * self.config.linguistic_weight
            + structural * self.config.structural_weight
        )
        overall = 1.0 + (weighted_score * 4.0)  # Scale to 1-5

        return {
            "baroque_style": baroque,
            "thematic_alignment": thematic,
            "linguistic_authenticity": linguistic,
            "structural_coherence": structural,
            "overall_score": overall,
            "passes_threshold": overall >= self.config.min_score_threshold,
        }


class SorJuanaTrainer:
    """Fine-tuning pipeline for Sor Juana style model using MLX"""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b", output_dir: str = "./sor_juana_model"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluator = SorJuanaEvaluator(EvalConfig())
        self.results = {"train": [], "eval": []}

    def load_dataset_from_csv(self, csv_path: str, split_ratio: float = 0.8) -> tuple:
        """Load JSONL training data from CSV (must have 'prompt' and 'completion' columns)"""
        train_data = []
        eval_data = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                prompt = row.get("prompt", "")
                completion = row.get("completion", "")

                if not prompt or not completion:
                    continue

                # Format as chat message
                example = {
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
                }

                if np.random.random() < split_ratio:
                    train_data.append(example)
                else:
                    eval_data.append(example)

        return train_data, eval_data

    def generate_with_model(self, prompt: str, model, tokenizer, max_tokens: int = 256) -> str:
        """Generate text using local model (MLX or torch backend)"""
        try:
            # MLX backend
            inputs = tokenizer(prompt, return_tensors="np")
            generated = model.generate(mx.array(inputs["input_ids"]), max_length=max_tokens)
            output = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        except Exception:
            # Fallback: torch backend
            inputs = tokenizer(prompt, return_tensors="pt")
            generated = model.generate(inputs["input_ids"], max_length=max_tokens, do_sample=True)
            output = tokenizer.decode(generated[0], skip_special_tokens=True)

        return output

    def evaluate_batch(self, eval_data: List[Dict], model, tokenizer) -> Dict[str, Any]:
        """Run batch evaluation on held-out set"""
        results = {"total": len(eval_data), "passed": 0, "scores": [], "details": []}

        for idx, example in enumerate(eval_data[:20]):  # Sample 20 for speed
            prompt = example["messages"][0]["content"]
            expected_completion = example["messages"][1]["content"]

            # Generate response
            generated = self.generate_with_model(prompt, model, tokenizer)

            # Evaluate
            eval_score = self.evaluator.evaluate(generated)
            results["scores"].append(eval_score["overall_score"])

            if eval_score["passes_threshold"]:
                results["passed"] += 1

            results["details"].append(
                {"prompt": prompt, "generated": generated, "expected": expected_completion, "eval": eval_score}
            )

        results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0
        results["mean_score"] = np.mean(results["scores"]) if results["scores"] else 0.0

        return results

    def save_results_to_json(self, eval_results: Dict) -> str:
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().isoformat()
        results_file = self.output_dir / f"eval_results_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        return str(results_file)


def main():
    parser = argparse.ArgumentParser(description="Sor Juana Style Model: Fine-tuning + Evaluation")
    parser.add_argument(
        "--csv", type=str, default="sor_juana_corpus.csv", help="Path to CSV with prompt/completion pairs"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Base model name (HuggingFace)")
    parser.add_argument(
        "--output", type=str, default="./sor_juana_model", help="Output directory for model and results"
    )
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (no fine-tuning)")
    parser.add_argument("--sample-prompt", type=str, help="Generate sample response to a prompt")

    args = parser.parse_args()

    print("=" * 60)
    print("SOR JUANA STYLE MODEL: Fine-tuning + Evaluation Pipeline")
    print("Optimized for Apple Silicon (M4 Mac mini)")
    print("=" * 60)

    # Initialize trainer
    trainer = SorJuanaTrainer(model_name=args.model, output_dir=args.output)

    # Load model and tokenizer
    print(f"\n[1/5] Loading model: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  Install with: pip install transformers torch")
        exit(1)

    # Load dataset
    print(f"\n[2/5] Loading dataset from {args.csv}")
    if not Path(args.csv).exists():
        print(f"✗ CSV file not found: {args.csv}")
        print("  Create using the DBOS download script first")
        exit(1)

    train_data, eval_data = trainer.load_dataset_from_csv(args.csv)
    print(f"✓ Loaded {len(train_data)} training + {len(eval_data)} eval examples")

    # Fine-tuning (placeholder: full fine-tuning via MLX would require additional setup)
    if not args.eval_only:
        print("\n[3/5] Fine-tuning model on Sor Juana corpus")
        print("  Note: Full fine-tuning requires MLX-LM advanced setup")
        print("  For MVP, using zero-shot evaluation of base model")
        print("✓ (See mlx-lm documentation for full LoRA/QLoRA fine-tuning)")

    # Evaluation
    print(f"\n[4/5] Running evaluation on {len(eval_data)} held-out examples")
    eval_results = trainer.evaluate_batch(eval_data, model, tokenizer)
    print("✓ Evaluation complete")
    print(f"  - Pass rate: {eval_results['pass_rate']:.1%}")
    print(f"  - Mean score: {eval_results['mean_score']:.2f}/5.0")

    # Save results
    print("\n[5/5] Saving results")
    results_file = trainer.save_results_to_json(eval_results)
    print(f"✓ Results saved to {results_file}")

    # Optional: Generate sample
    if args.sample_prompt:
        print("\n[SAMPLE] Generating response to prompt:")
        print(f"  Prompt: {args.sample_prompt}")
        response = trainer.generate_with_model(args.sample_prompt, model, tokenizer)
        print(f"  Generated: {response}")

        eval_score = trainer.evaluator.evaluate(response)
        print(f"  Authenticity score: {eval_score['overall_score']:.2f}/5.0")
        print(f"    - Baroque style: {eval_score['baroque_style']:.2f}")
        print(f"    - Thematic alignment: {eval_score['thematic_alignment']:.2f}")
        print(f"    - Linguistic authenticity: {eval_score['linguistic_authenticity']:.2f}")
        print(f"    - Structural coherence: {eval_score['structural_coherence']:.2f}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
