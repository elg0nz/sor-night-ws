"""DBOS workflows for OpenAI model fine-tuning."""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dbos import DBOS, Queue
from .config import DBOS_CONFIG, DATA_DIR

# DBOS Configuration - must be at module level
DBOS(config=DBOS_CONFIG)

# Queue for training operations
training_queue = Queue("training_queue")

# Configuration
OUTPUT_DIR = DATA_DIR / "openai_training"
OUTPUT_DIR.mkdir(exist_ok=True)

FINE_TUNE_CONFIG = {
    "model": "gpt-4o-mini-2024-07-18",
    "n_epochs": 3,
    "batch_size": None,
    "learning_rate_multiplier": None,
}


@DBOS.step()
def transform_to_openai_format_step(
    input_file: str, output_file: str, max_examples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Transform corpus JSONL to OpenAI fine-tuning format.
    
    Args:
        input_file: Path to corpus JSONL file
        output_file: Path to save transformed JSONL
        max_examples: Limit number of examples
        
    Returns:
        Dictionary with transformation results
    """
    system_message = (
        "Eres Sor Juana Inés de la Cruz, poeta y pensadora del siglo XVII. "
        "Escribes en estilo barroco con profundidad teológica, filosófica y feminista. "
        "Tu lenguaje es elegante, culto y lleno de referencias clásicas."
    )
    
    examples_processed = 0
    
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if max_examples and examples_processed >= max_examples:
                break
                
            item = json.loads(line)
            text = item.get("text", "").strip()
            metadata = item.get("metadata", {})
            
            # Skip very short texts
            if len(text) < 50:
                continue
                
            # Create genre-specific user prompts
            genre = metadata.get("genre", "poetry")
            if "poetry" in genre.lower() or "poesia" in genre.lower():
                user_prompt = "Escribe un poema o verso en tu característico estilo barroco."
            elif "letter" in genre.lower() or "carta" in genre.lower():
                user_prompt = "Escribe una carta o reflexión en tu estilo literario."
            else:
                user_prompt = "Escribe en tu característico estilo barroco y filosófico."
                
            # Create OpenAI training example
            training_example = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": text},
                ]
            }
            
            outfile.write(json.dumps(training_example, ensure_ascii=False) + "\n")
            examples_processed += 1
            
    return {
        "input_file": input_file,
        "output_file": output_file,
        "examples_processed": examples_processed,
    }


@DBOS.step()
def upload_file_to_openai_step(file_path: str, purpose: str = "fine-tune") -> str:
    """
    Upload a file to OpenAI.
    
    Args:
        file_path: Path to file to upload
        purpose: File purpose
        
    Returns:
        File ID from OpenAI
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    client = OpenAI(api_key=api_key)
    
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
        
    return response.id


@DBOS.step()
def create_fine_tuning_job_step(
    training_file_id: str,
    validation_file_id: Optional[str] = None,
    model: str = "gpt-4o-mini-2024-07-18",
    suffix: Optional[str] = "sor-juana",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a fine-tuning job on OpenAI.
    
    Args:
        training_file_id: ID of uploaded training file
        validation_file_id: Optional ID of validation file
        model: Base model to fine-tune
        suffix: Suffix for the fine-tuned model name
        hyperparameters: Optional hyperparameters
        
    Returns:
        Dictionary with job information
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    client = OpenAI(api_key=api_key)
    
    job_params = {
        "training_file": training_file_id,
        "model": model,
    }
    
    if validation_file_id:
        job_params["validation_file"] = validation_file_id
        
    if suffix:
        job_params["suffix"] = suffix
        
    if hyperparameters:
        job_params["hyperparameters"] = hyperparameters
        
    response = client.fine_tuning.jobs.create(**job_params)
    
    return {
        "job_id": response.id,
        "status": response.status,
        "model": response.model,
        "created_at": response.created_at,
    }


@DBOS.step()
def get_job_status_step(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a fine-tuning job.
    
    Args:
        job_id: Fine-tuning job ID
        
    Returns:
        Dictionary with job status information
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    client = OpenAI(api_key=api_key)
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    result = {
        "job_id": job.id,
        "status": job.status,
        "model": job.model,
        "created_at": job.created_at,
    }
    
    if job.fine_tuned_model:
        result["fine_tuned_model"] = job.fine_tuned_model
        
    if job.finished_at:
        result["finished_at"] = job.finished_at
        
    if job.error:
        result["error"] = str(job.error)
        
    return result


@DBOS.step()
def list_job_events_step(job_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    List recent events for a fine-tuning job.
    
    Args:
        job_id: Fine-tuning job ID
        limit: Maximum number of events to retrieve
        
    Returns:
        List of event dictionaries
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    client = OpenAI(api_key=api_key)
    
    try:
        events = client.fine_tuning.jobs.list_events(job_id, limit=limit)
        return [{"created_at": e.created_at, "message": e.message} for e in events.data]
    except Exception:
        return []


@DBOS.step()
def save_model_info_step(model_info: Dict[str, Any]) -> str:
    """
    Save fine-tuned model information to a file.
    
    Args:
        model_info: Dictionary with model information
        
    Returns:
        Path to saved file
    """
    model_info_file = OUTPUT_DIR / "fine_tuned_model.json"
    
    with open(model_info_file, "w") as f:
        json.dump(model_info, f, indent=2)
        
    return str(model_info_file)


@DBOS.workflow()
def prepare_training_data_workflow(
    train_file: str, eval_file: str, max_examples: Optional[int] = None
) -> Dict[str, Any]:
    """
    DBOS workflow to prepare training data for OpenAI fine-tuning.
    
    Args:
        train_file: Path to training JSONL file
        eval_file: Path to evaluation JSONL file
        max_examples: Optional limit on number of examples
        
    Returns:
        Dictionary with preparation results
    """
    openai_train_file = str(OUTPUT_DIR / "train_openai.jsonl")
    openai_eval_file = str(OUTPUT_DIR / "eval_openai.jsonl")
    
    # Transform training data
    train_handle = training_queue.enqueue(
        transform_to_openai_format_step, train_file, openai_train_file, max_examples
    )
    train_result = train_handle.get_result()
    
    # Transform eval data
    eval_handle = training_queue.enqueue(
        transform_to_openai_format_step, eval_file, openai_eval_file, max_examples
    )
    eval_result = eval_handle.get_result()
    
    return {
        "train": train_result,
        "eval": eval_result,
        "train_file": openai_train_file,
        "eval_file": openai_eval_file,
    }


@DBOS.workflow()
def start_fine_tuning_workflow(
    train_file: str,
    eval_file: str,
    model: Optional[str] = None,
    suffix: Optional[str] = "sor-juana",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    DBOS workflow to start OpenAI fine-tuning.
    
    Args:
        train_file: Path to training file
        eval_file: Path to evaluation file
        model: Base model to fine-tune
        suffix: Suffix for fine-tuned model name
        hyperparameters: Optional hyperparameters
        
    Returns:
        Dictionary with job information
    """
    if model is None:
        model = FINE_TUNE_CONFIG["model"]
        
    if hyperparameters is None:
        hyperparameters = {}
        if FINE_TUNE_CONFIG.get("n_epochs"):
            hyperparameters["n_epochs"] = FINE_TUNE_CONFIG["n_epochs"]
        if FINE_TUNE_CONFIG.get("batch_size"):
            hyperparameters["batch_size"] = FINE_TUNE_CONFIG["batch_size"]
        if FINE_TUNE_CONFIG.get("learning_rate_multiplier"):
            hyperparameters["learning_rate_multiplier"] = FINE_TUNE_CONFIG["learning_rate_multiplier"]
    
    # Upload files
    train_upload_handle = training_queue.enqueue(upload_file_to_openai_step, train_file)
    eval_upload_handle = training_queue.enqueue(upload_file_to_openai_step, eval_file)
    
    train_file_id = train_upload_handle.get_result()
    eval_file_id = eval_upload_handle.get_result()
    
    # Create fine-tuning job
    job_result = create_fine_tuning_job_step(
        train_file_id, eval_file_id, model, suffix, hyperparameters if hyperparameters else None
    )
    
    return {
        "train_file_id": train_file_id,
        "eval_file_id": eval_file_id,
        **job_result,
    }


@DBOS.workflow()
def monitor_fine_tuning_workflow(job_id: str, poll_interval: int = 60) -> Dict[str, Any]:
    """
    DBOS workflow to monitor a fine-tuning job until completion.
    
    Args:
        job_id: Fine-tuning job ID
        poll_interval: Seconds between status checks
        
    Returns:
        Dictionary with final job status
    """
    while True:
        # Get job status
        status_result = get_job_status_step(job_id)
        
        status = status_result["status"]
        
        # Check if job is complete
        if status in ["succeeded", "failed", "cancelled"]:
            # Save model info if succeeded
            if status == "succeeded":
                save_model_info_step(status_result)
                
            return status_result
            
        # Sleep before next check
        DBOS.sleep(poll_interval)


@DBOS.workflow()
def full_fine_tuning_workflow(
    train_jsonl: Optional[str] = None,
    eval_jsonl: Optional[str] = None,
    max_examples: Optional[int] = None,
    monitor: bool = True,
) -> Dict[str, Any]:
    """
    Complete DBOS workflow for fine-tuning: prepare data, upload, create job, and optionally monitor.
    
    Args:
        train_jsonl: Path to training JSONL (default: data/train.jsonl)
        eval_jsonl: Path to eval JSONL (default: data/eval.jsonl)
        max_examples: Optional limit on examples
        monitor: Whether to monitor the job until completion
        
    Returns:
        Dictionary with complete workflow results
    """
    # Use defaults if not provided
    if train_jsonl is None:
        train_jsonl = str(DATA_DIR / "train.jsonl")
    if eval_jsonl is None:
        eval_jsonl = str(DATA_DIR / "eval.jsonl")
        
    # Step 1: Prepare training data
    prep_result = prepare_training_data_workflow(train_jsonl, eval_jsonl, max_examples)
    
    # Step 2: Start fine-tuning
    job_result = start_fine_tuning_workflow(
        prep_result["train_file"], prep_result["eval_file"]
    )
    
    # Step 3: Optionally monitor
    final_result = {
        "preparation": prep_result,
        "job": job_result,
    }
    
    if monitor:
        monitor_result = monitor_fine_tuning_workflow(job_result["job_id"])
        final_result["monitor"] = monitor_result
        
    return final_result


@DBOS.workflow()
def get_job_info_workflow(job_id: str) -> Dict[str, Any]:
    """
    DBOS workflow to get information about a fine-tuning job.
    
    Args:
        job_id: Fine-tuning job ID
        
    Returns:
        Dictionary with job information and recent events
    """
    # Get job status
    status_handle = training_queue.enqueue(get_job_status_step, job_id)
    status_result = status_handle.get_result()
    
    # Get recent events
    events_handle = training_queue.enqueue(list_job_events_step, job_id, 10)
    events = events_handle.get_result()
    
    return {
        "status": status_result,
        "recent_events": events,
    }

