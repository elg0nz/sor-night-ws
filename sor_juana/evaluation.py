"""DBOS workflows for model evaluation and testing."""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dbos import DBOS, Queue
from .config import DBOS_CONFIG, DATA_DIR

# DBOS Configuration - must be at module level
DBOS(config=DBOS_CONFIG)

# Queue for evaluation operations
eval_queue = Queue("eval_queue")

# Test prompts in Spanish
TEST_PROMPTS = [
    {
        "name": "Soneto sobre el conocimiento",
        "prompt": "Escribe un soneto sobre la búsqueda del conocimiento.",
    },
    {
        "name": "Reflexión sobre el amor",
        "prompt": "Escribe una reflexión poética sobre la naturaleza del amor.",
    },
    {
        "name": "Carta filosófica",
        "prompt": "Escribe una carta breve expresando ideas filosóficas sobre la libertad.",
    },
    {
        "name": "Verso sobre la fe",
        "prompt": "Escribe versos sobre la relación entre fe y razón.",
    },
    {
        "name": "Todos debemos de programar",
        "prompt": "Escribe una reflexión filosófica sobre la necesidad de programar, y como las mujeres deben de estar involucradas en el desarrollo de software",
    },
]


@DBOS.step()
def load_model_id_step() -> str:
    """
    Load model ID from the saved training output.
    
    Returns:
        Model ID string
    """
    data_dir = DATA_DIR / "openai_training"
    model_file = data_dir / "fine_tuned_model.json"
    
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}. Run training first."
        )
        
    with open(model_file, "r") as f:
        data = json.load(f)
        
    model_id = data.get("fine_tuned_model") or data.get("model_id")
    if not model_id:
        raise ValueError("No model ID found in saved file")
        
    return model_id


@DBOS.step()
def generate_text_step(
    model_id: str,
    prompt: str,
    system_message: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.8,
) -> str:
    """
    Generate text using a model.
    
    Args:
        model_id: Model ID to use
        prompt: User prompt
        system_message: Optional system message
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    client = OpenAI(api_key=api_key)
    
    if system_message is None:
        system_message = (
            "Eres Sor Juana Inés de la Cruz, poeta y pensadora del siglo XVII. "
            "Escribes en estilo barroco con profundidad teológica, filosófica y sobre todo feminista."
        )
        
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return response.choices[0].message.content


@DBOS.step()
def evaluate_baroque_style_step(text: str) -> Dict[str, float]:
    """
    Evaluate text for Baroque rhetorical features.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with style scores
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
            
    return {
        "baroque_score": score,
        "keywords_found": keywords_found,
        "avg_sentence_length": float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0,
    }


@DBOS.step()
def evaluate_thematic_alignment_step(text: str) -> Dict[str, float]:
    """
    Evaluate alignment with Sor Juana's major themes.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with theme scores
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
        
    return {
        "thematic_score": min(1.0, score),
        "themes_found": themes_found,
    }


@DBOS.step()
def evaluate_text_step(text: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation of generated text.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with all evaluation scores
    """
    # Get baroque style evaluation
    baroque_eval = evaluate_baroque_style_step(text)
    
    # Get thematic alignment
    thematic_eval = evaluate_thematic_alignment_step(text)
    
    # Calculate overall score (1-5 scale)
    baroque_score = baroque_eval["baroque_score"]
    thematic_score = thematic_eval["thematic_score"]
    
    # Weighted average
    weighted_score = (baroque_score * 0.5 + thematic_score * 0.5)
    overall = 1.0 + (weighted_score * 4.0)  # Scale to 1-5
    
    return {
        "baroque_style": baroque_score,
        "thematic_alignment": thematic_score,
        "overall_score": overall,
        "passes_threshold": overall >= 3.0,
        "details": {
            **baroque_eval,
            **thematic_eval,
        },
    }


@DBOS.workflow()
def test_model_with_prompt_workflow(
    prompt: str,
    model_id: Optional[str] = None,
    evaluate: bool = True,
) -> Dict[str, Any]:
    """
    DBOS workflow to test a model with a specific prompt.
    
    Args:
        prompt: User prompt
        model_id: Model ID to use (loads from file if not provided)
        evaluate: Whether to evaluate the generated text
        
    Returns:
        Dictionary with generated text and evaluation
    """
    # Load model ID if not provided
    if model_id is None:
        model_id = load_model_id_step()
        
    # Generate text
    generated_text = generate_text_step(model_id, prompt)
    
    result = {
        "model_id": model_id,
        "prompt": prompt,
        "generated_text": generated_text,
    }
    
    # Evaluate if requested
    if evaluate:
        eval_result = evaluate_text_step(generated_text)
        result["evaluation"] = eval_result
        
    return result


@DBOS.workflow()
def test_model_with_prompts_workflow(
    prompts: Optional[List[Dict[str, str]]] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    DBOS workflow to test a model with multiple prompts.
    
    Args:
        prompts: List of prompt dictionaries (uses defaults if not provided)
        model_id: Model ID to use (loads from file if not provided)
        
    Returns:
        Dictionary with all test results
    """
    # Use default prompts if not provided
    if prompts is None:
        prompts = TEST_PROMPTS
        
    # Load model ID if not provided
    if model_id is None:
        model_id = load_model_id_step()
        
    # Test each prompt
    results = []
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        name = prompt_data.get("name", "Unnamed")
        
        # Enqueue test for each prompt
        test_handle = eval_queue.enqueue(
            test_model_with_prompt_workflow, prompt, model_id, True
        )
        test_result = test_handle.get_result()
        
        results.append({
            "name": name,
            **test_result,
        })
        
    # Calculate summary statistics
    scores = [r["evaluation"]["overall_score"] for r in results if "evaluation" in r]
    passed = sum(1 for r in results if r.get("evaluation", {}).get("passes_threshold", False))
    
    return {
        "model_id": model_id,
        "total_tests": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "results": results,
    }


@DBOS.workflow()
def compare_models_workflow(
    prompt: str,
    fine_tuned_model: Optional[str] = None,
    base_model: str = "gpt-4o-mini-2024-07-18",
) -> Dict[str, Any]:
    """
    DBOS workflow to compare fine-tuned model with base model.
    
    Args:
        prompt: User prompt
        fine_tuned_model: Fine-tuned model ID (loads from file if not provided)
        base_model: Base model ID for comparison
        
    Returns:
        Dictionary with comparison results
    """
    # Load fine-tuned model ID if not provided
    if fine_tuned_model is None:
        fine_tuned_model = load_model_id_step()
        
    # Generate with fine-tuned model
    fine_tuned_handle = eval_queue.enqueue(
        generate_text_step, fine_tuned_model, prompt
    )
    fine_tuned_text = fine_tuned_handle.get_result()
    
    # Generate with base model
    base_handle = eval_queue.enqueue(
        generate_text_step, base_model, prompt
    )
    base_text = base_handle.get_result()
    
    # Evaluate both
    fine_tuned_eval_handle = eval_queue.enqueue(
        evaluate_text_step, fine_tuned_text
    )
    base_eval_handle = eval_queue.enqueue(
        evaluate_text_step, base_text
    )
    
    fine_tuned_eval = fine_tuned_eval_handle.get_result()
    base_eval = base_eval_handle.get_result()
    
    return {
        "prompt": prompt,
        "fine_tuned": {
            "model_id": fine_tuned_model,
            "generated_text": fine_tuned_text,
            "evaluation": fine_tuned_eval,
        },
        "base": {
            "model_id": base_model,
            "generated_text": base_text,
            "evaluation": base_eval,
        },
        "score_difference": fine_tuned_eval["overall_score"] - base_eval["overall_score"],
    }


@DBOS.workflow()
def save_test_results_workflow(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    DBOS workflow to save test results to a file.
    
    Args:
        results: Test results dictionary
        output_file: Optional output file path
        
    Returns:
        Path to saved file
    """
    from datetime import datetime
    
    if output_file is None:
        timestamp = datetime.now().isoformat()
        output_file = str(DATA_DIR / f"test_results_{timestamp}.json")
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    return output_file

