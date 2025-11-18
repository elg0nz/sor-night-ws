"""DBOS workflows for local MLX-based model training on Apple Silicon."""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dbos import DBOS, Queue
from .config import DBOS_CONFIG, DATA_DIR

# DBOS Configuration - must be at module level
DBOS(config=DBOS_CONFIG)

# Queue for local training operations
local_training_queue = Queue("local_training_queue")


@DBOS.step()
def load_dataset_from_csv_step(
    csv_path: str, split_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    Load training dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file with prompt/completion columns
        split_ratio: Train/eval split ratio
        
    Returns:
        Dictionary with train and eval data
    """
    import numpy as np
    
    train_data = []
    eval_data = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "")
            completion = row.get("completion", "")
            
            if not prompt or not completion:
                continue
                
            # Format as chat message
            example = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }
            
            if np.random.random() < split_ratio:
                train_data.append(example)
            else:
                eval_data.append(example)
                
    return {
        "train_data": train_data,
        "eval_data": eval_data,
        "train_count": len(train_data),
        "eval_count": len(eval_data),
    }


@DBOS.step()
def load_model_step(model_name: str) -> Dict[str, Any]:
    """
    Load model and tokenizer.
    
    Args:
        model_name: Model name from HuggingFace
        
    Returns:
        Dictionary with model loading status
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        return {
            "model_name": model_name,
            "status": "success",
            "model_type": type(model).__name__,
        }
    except ImportError as e:
        return {
            "model_name": model_name,
            "status": "error",
            "error": f"Missing dependencies: {str(e)}",
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "status": "error",
            "error": str(e),
        }


@DBOS.step()
def score_baroque_style_step(text: str) -> float:
    """
    Score text for Baroque rhetorical features.
    
    Args:
        text: Text to score
        
    Returns:
        Baroque style score (0-1)
    """
    import numpy as np
    
    baroque_keywords = {
        "conceptos": 0.8,
        "agudeza": 0.9,
        "paradoja": 0.85,
        "metáfora": 0.7,
        "antítesis": 0.8,
        "hipérbole": 0.75,
    }
    
    text_lower = text.lower()
    score = 0.0
    keywords_found = 0
    
    for keyword, weight in baroque_keywords.items():
        if keyword in text_lower:
            score += weight
            keywords_found += 1
            
    # Normalize and add structural complexity bonus
    if keywords_found > 0:
        score = min(1.0, (score / len(baroque_keywords)) * 1.5)
        
    # Bonus for longer, complex sentences (typical of Baroque)
    sentences = [s for s in text.split(".") if s.strip()]
    if sentences:
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        if avg_sentence_length > 15:
            score = min(1.0, score + 0.15)
            
    return float(score)


@DBOS.step()
def score_thematic_alignment_step(text: str) -> float:
    """
    Score alignment with Sor Juana's major themes.
    
    Args:
        text: Text to score
        
    Returns:
        Thematic alignment score (0-1)
    """
    feminist_themes = {
        "educación": 0.9,
        "mujer": 0.8,
        "ingenio": 0.85,
        "saber": 0.8,
        "derecho": 0.85,
        "injusticia": 0.7,
    }
    
    theological_terms = {
        "alma": 0.7,
        "divino": 0.75,
        "sacramento": 0.85,
        "gracia": 0.8,
        "penitencia": 0.75,
    }
    
    text_lower = text.lower()
    score = 0.0
    themes_found = 0
    
    # Check feminist/intellectual themes
    for theme, weight in feminist_themes.items():
        if theme in text_lower:
            score += weight
            themes_found += 1
            
    # Check theological themes
    for term, weight in theological_terms.items():
        if term in text_lower:
            score += weight
            themes_found += 1
            
    if themes_found > 0:
        score = score / (len(feminist_themes) + len(theological_terms))
        
    return float(min(1.0, score))


@DBOS.step()
def score_linguistic_authenticity_step(text: str) -> float:
    """
    Score period-appropriate language usage.
    
    Args:
        text: Text to score
        
    Returns:
        Linguistic authenticity score (0-1)
    """
    period_vocabulary = {
        "vos": 0.8,
        "vuestra merced": 0.9,
        "fiera": 0.7,
        "grosero": 0.7,
        "tinieblas": 0.75,
    }
    
    text_lower = text.lower()
    score = 0.0
    features_found = 0
    
    for vocab, weight in period_vocabulary.items():
        if vocab in text_lower:
            score += weight
            features_found += 1
            
    # Penalize modern language
    modern_terms = ["ai", "computadora", "internet", "moderno"]
    for term in modern_terms:
        if term in text_lower:
            score -= 0.2
            
    if features_found > 0:
        score = score / len(period_vocabulary)
        
    return float(max(0.0, min(1.0, score)))


@DBOS.step()
def score_structural_coherence_step(text: str) -> float:
    """
    Score structural integrity (sonnet, décima, coherence).
    
    Args:
        text: Text to score
        
    Returns:
        Structural coherence score (0-1)
    """
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
        
    return float(min(1.0, score))


@DBOS.step()
def evaluate_text_comprehensive_step(text: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation of text for Sor Juana style.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with all evaluation scores
    """
    baroque = score_baroque_style_step(text)
    thematic = score_thematic_alignment_step(text)
    linguistic = score_linguistic_authenticity_step(text)
    structural = score_structural_coherence_step(text)
    
    # Weighted average to 1.0-5.0 scale
    weighted_score = (
        baroque * 0.25
        + thematic * 0.25
        + linguistic * 0.25
        + structural * 0.25
    )
    overall = 1.0 + (weighted_score * 4.0)  # Scale to 1-5
    
    return {
        "baroque_style": baroque,
        "thematic_alignment": thematic,
        "linguistic_authenticity": linguistic,
        "structural_coherence": structural,
        "overall_score": overall,
        "passes_threshold": overall >= 3.0,
    }


@DBOS.step()
def generate_with_model_step(
    prompt: str, model_name: str, max_tokens: int = 256
) -> str:
    """
    Generate text using local model.
    
    Args:
        prompt: Input prompt
        model_name: Model name
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Try MLX backend first
        try:
            import mlx.core as mx
            
            inputs = tokenizer(prompt, return_tensors="np")
            generated = model.generate(
                mx.array(inputs["input_ids"]), max_length=max_tokens
            )
            output = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        except Exception:
            # Fallback: torch backend
            inputs = tokenizer(prompt, return_tensors="pt")
            generated = model.generate(
                inputs["input_ids"], max_length=max_tokens, do_sample=True
            )
            output = tokenizer.decode(generated[0], skip_special_tokens=True)
            
        return output
    except Exception as e:
        return f"Error generating text: {str(e)}"


@DBOS.step()
def evaluate_batch_step(
    eval_data: List[Dict], model_name: str, sample_size: int = 20
) -> Dict[str, Any]:
    """
    Run batch evaluation on held-out set.
    
    Args:
        eval_data: List of evaluation examples
        model_name: Model name
        sample_size: Number of examples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    import numpy as np
    
    results = {
        "total": min(len(eval_data), sample_size),
        "passed": 0,
        "scores": [],
        "details": [],
    }
    
    for idx, example in enumerate(eval_data[:sample_size]):
        prompt = example["messages"][0]["content"]
        expected_completion = example["messages"][1]["content"]
        
        # Generate response
        generated = generate_with_model_step(prompt, model_name)
        
        # Evaluate
        eval_score = evaluate_text_comprehensive_step(generated)
        results["scores"].append(eval_score["overall_score"])
        
        if eval_score["passes_threshold"]:
            results["passed"] += 1
            
        results["details"].append({
            "prompt": prompt,
            "generated": generated,
            "expected": expected_completion,
            "eval": eval_score,
        })
        
    results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0
    results["mean_score"] = float(np.mean(results["scores"])) if results["scores"] else 0.0
    
    return results


@DBOS.step()
def save_results_to_json_step(eval_results: Dict, output_dir: str) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
        eval_results: Evaluation results dictionary
        output_dir: Output directory
        
    Returns:
        Path to saved results file
    """
    timestamp = datetime.now().isoformat()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    results_file = output_path / f"eval_results_{timestamp}.json"
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
    return str(results_file)


@DBOS.workflow()
def load_and_prepare_dataset_workflow(
    csv_path: str, split_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    DBOS workflow to load and prepare training dataset.
    
    Args:
        csv_path: Path to CSV file
        split_ratio: Train/eval split ratio
        
    Returns:
        Dictionary with dataset information
    """
    dataset_handle = local_training_queue.enqueue(
        load_dataset_from_csv_step, csv_path, split_ratio
    )
    dataset = dataset_handle.get_result()
    
    return dataset


@DBOS.workflow()
def evaluate_model_workflow(
    model_name: str,
    eval_data: List[Dict],
    output_dir: str,
    sample_size: int = 20,
) -> Dict[str, Any]:
    """
    DBOS workflow to evaluate a model on held-out data.
    
    Args:
        model_name: Model name to evaluate
        eval_data: Evaluation dataset
        output_dir: Directory to save results
        sample_size: Number of examples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Run batch evaluation
    eval_handle = local_training_queue.enqueue(
        evaluate_batch_step, eval_data, model_name, sample_size
    )
    eval_results = eval_handle.get_result()
    
    # Save results
    save_handle = local_training_queue.enqueue(
        save_results_to_json_step, eval_results, output_dir
    )
    results_file = save_handle.get_result()
    
    eval_results["results_file"] = results_file
    
    return eval_results


@DBOS.workflow()
def local_training_pipeline_workflow(
    csv_path: str,
    model_name: str = "meta-llama/Llama-2-7b",
    output_dir: str = "./sor_juana_model",
    eval_only: bool = True,
    sample_size: int = 20,
) -> Dict[str, Any]:
    """
    Complete DBOS workflow for local model training and evaluation.
    
    Args:
        csv_path: Path to CSV with prompt/completion pairs
        model_name: Base model name
        output_dir: Output directory for results
        eval_only: Whether to only evaluate (no fine-tuning)
        sample_size: Number of examples to evaluate
        
    Returns:
        Dictionary with pipeline results
    """
    # Step 1: Load dataset
    dataset = load_and_prepare_dataset_workflow(csv_path)
    
    # Step 2: Load model (check if it's available)
    model_status_handle = local_training_queue.enqueue(
        load_model_step, model_name
    )
    model_status = model_status_handle.get_result()
    
    if model_status["status"] == "error":
        return {
            "status": "error",
            "error": model_status["error"],
            "dataset": dataset,
        }
        
    # Step 3: Fine-tuning (placeholder for now)
    if not eval_only:
        # Note: Full fine-tuning requires MLX-LM advanced setup
        # For MVP, we're using zero-shot evaluation
        fine_tuning_result = {
            "status": "skipped",
            "message": "Full fine-tuning requires MLX-LM advanced setup. See mlx-lm documentation.",
        }
    else:
        fine_tuning_result = {"status": "skipped", "message": "Eval-only mode"}
        
    # Step 4: Evaluation
    eval_results = evaluate_model_workflow(
        model_name, dataset["eval_data"], output_dir, sample_size
    )
    
    return {
        "status": "success",
        "model_name": model_name,
        "dataset": {
            "train_count": dataset["train_count"],
            "eval_count": dataset["eval_count"],
        },
        "fine_tuning": fine_tuning_result,
        "evaluation": {
            "pass_rate": eval_results["pass_rate"],
            "mean_score": eval_results["mean_score"],
            "total_evaluated": eval_results["total"],
            "results_file": eval_results["results_file"],
        },
    }


@DBOS.workflow()
def generate_sample_workflow(
    prompt: str, model_name: str = "meta-llama/Llama-2-7b"
) -> Dict[str, Any]:
    """
    DBOS workflow to generate a sample response and evaluate it.
    
    Args:
        prompt: Input prompt
        model_name: Model name to use
        
    Returns:
        Dictionary with generated text and evaluation
    """
    # Generate text
    generated_handle = local_training_queue.enqueue(
        generate_with_model_step, prompt, model_name
    )
    generated = generated_handle.get_result()
    
    # Evaluate
    eval_handle = local_training_queue.enqueue(
        evaluate_text_comprehensive_step, generated
    )
    evaluation = eval_handle.get_result()
    
    return {
        "prompt": prompt,
        "generated": generated,
        "evaluation": evaluation,
    }

