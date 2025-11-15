#!/usr/bin/env python3
"""
OpenAI Fine-tuning Script for Sor Juana Writing Style

This script handles the complete pipeline for fine-tuning an OpenAI model
to write in the style of Sor Juana Inés de la Cruz.

Steps:
1. Load and transform the corpus data into OpenAI's format
2. Upload training and validation files to OpenAI
3. Create a fine-tuning job
4. Monitor the training progress
5. Save the fine-tuned model ID for later use

Requirements:
- OpenAI API key set as OPENAI_API_KEY environment variable
- Install: pip install openai rich
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install openai rich")
    sys.exit(1)

# Initialize Rich console for beautiful output
console = Console()

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"
OUTPUT_DIR = DATA_DIR / "openai_training"
OUTPUT_DIR.mkdir(exist_ok=True)

# OpenAI fine-tuning configuration
FINE_TUNE_CONFIG = {
    "model": "gpt-4o-mini-2024-07-18",  # Base model to fine-tune
    "n_epochs": 3,  # Number of training epochs
    "batch_size": None,  # Auto-select batch size
    "learning_rate_multiplier": None,  # Auto-select learning rate
}


def validate_api_key() -> OpenAI:
    """
    Validate that OpenAI API key is set and create client.

    Returns:
        OpenAI client instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]❌ Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("\nSet it with:")
        console.print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    console.print("[green]✓[/green] API key found")
    return OpenAI(api_key=api_key)


def transform_to_openai_format(input_file: Path, output_file: Path, max_examples: Optional[int] = None) -> int:
    """
    Transform corpus JSONL to OpenAI fine-tuning format.

    The corpus format is:
        {"text": "...", "metadata": {...}}

    OpenAI format for chat models:
        {"messages": [
            {"role": "system", "content": "You are Sor Juana Inés de la Cruz..."},
            {"role": "user", "content": "Write about [topic]"},
            {"role": "assistant", "content": "actual text from corpus"}
        ]}

    Args:
        input_file: Path to corpus JSONL file
        output_file: Path to save transformed JSONL
        max_examples: Limit number of examples (for testing)

    Returns:
        Number of examples processed
    """
    console.print(f"[cyan]Transforming {input_file.name}...[/cyan]")

    system_message = (
        "Eres Sor Juana Inés de la Cruz, poeta y pensadora del siglo XVII. "
        "Escribes en estilo barroco con profundidad teológica, filosófica y feminista. "
        "Tu lenguaje es elegante, culto y lleno de referencias clásicas."
    )

    examples_processed = 0

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            if max_examples and examples_processed >= max_examples:
                break

            item = json.loads(line)
            text = item.get("text", "").strip()
            metadata = item.get("metadata", {})

            # Skip very short texts (likely not meaningful content)
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

    console.print(f"[green]✓[/green] Processed {examples_processed} examples")
    return examples_processed


def upload_file(client: OpenAI, file_path: Path, purpose: str = "fine-tune") -> str:
    """
    Upload a file to OpenAI.

    Args:
        client: OpenAI client
        file_path: Path to file to upload
        purpose: File purpose (usually "fine-tune")

    Returns:
        File ID from OpenAI
    """
    console.print(f"[cyan]Uploading {file_path.name}...[/cyan]")

    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)

    console.print(f"[green]✓[/green] Uploaded: {response.id}")
    return response.id


def create_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    validation_file_id: Optional[str] = None,
    model: str = "gpt-4o-mini-2024-07-18",
    suffix: Optional[str] = "sor-juana",
) -> str:
    """
    Create a fine-tuning job on OpenAI.

    Args:
        client: OpenAI client
        training_file_id: ID of uploaded training file
        validation_file_id: Optional ID of validation file
        model: Base model to fine-tune
        suffix: Suffix for the fine-tuned model name

    Returns:
        Fine-tuning job ID
    """
    console.print("[cyan]Creating fine-tuning job...[/cyan]")

    job_params = {
        "training_file": training_file_id,
        "model": model,
    }

    if validation_file_id:
        job_params["validation_file"] = validation_file_id

    if suffix:
        job_params["suffix"] = suffix

    # Add hyperparameters if specified
    hyperparameters = {}
    if FINE_TUNE_CONFIG.get("n_epochs"):
        hyperparameters["n_epochs"] = FINE_TUNE_CONFIG["n_epochs"]
    if FINE_TUNE_CONFIG.get("batch_size"):
        hyperparameters["batch_size"] = FINE_TUNE_CONFIG["batch_size"]
    if FINE_TUNE_CONFIG.get("learning_rate_multiplier"):
        hyperparameters["learning_rate_multiplier"] = FINE_TUNE_CONFIG["learning_rate_multiplier"]

    if hyperparameters:
        job_params["hyperparameters"] = hyperparameters

    response = client.fine_tuning.jobs.create(**job_params)

    console.print(f"[green]✓[/green] Job created: {response.id}")
    console.print(f"   Status: {response.status}")
    console.print(f"   Model: {response.model}")

    return response.id


def monitor_fine_tuning_job(client: OpenAI, job_id: str, poll_interval: int = 60):
    """
    Monitor a fine-tuning job until completion.

    Args:
        client: OpenAI client
        job_id: Fine-tuning job ID
        poll_interval: Seconds between status checks
    """
    console.print(f"\n[cyan]Monitoring job {job_id}...[/cyan]")
    console.print(f"Checking every {poll_interval} seconds (Ctrl+C to stop monitoring)\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training in progress...", total=None)

        try:
            while True:
                job = client.fine_tuning.jobs.retrieve(job_id)
                status = job.status

                # Update progress description
                progress.update(task, description=f"Status: {status}")

                # Check if job is complete
                if status == "succeeded":
                    progress.stop()
                    console.print("\n[green]✓ Fine-tuning completed successfully![/green]")
                    console.print(f"[green]Fine-tuned model: {job.fine_tuned_model}[/green]")

                    # Save model info
                    model_info_file = OUTPUT_DIR / "fine_tuned_model.json"
                    with open(model_info_file, "w") as f:
                        json.dump(
                            {
                                "job_id": job_id,
                                "model_id": job.fine_tuned_model,
                                "status": status,
                                "created_at": job.created_at,
                                "finished_at": job.finished_at,
                            },
                            f,
                            indent=2,
                        )

                    console.print(f"[green]Model info saved to: {model_info_file}[/green]")
                    return job.fine_tuned_model

                elif status == "failed":
                    progress.stop()
                    console.print("\n[red]❌ Fine-tuning failed[/red]")
                    if job.error:
                        console.print(f"[red]Error: {job.error}[/red]")
                    sys.exit(1)

                elif status == "cancelled":
                    progress.stop()
                    console.print("\n[yellow]⚠ Fine-tuning was cancelled[/yellow]")
                    sys.exit(1)

                # Wait before next check
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            progress.stop()
            console.print("\n[yellow]Monitoring stopped. Job is still running.[/yellow]")
            console.print(f"Check status with: job_id = '{job_id}'")
            return None


def display_summary(train_examples: int, eval_examples: int, job_id: str):
    """Display a summary table of the fine-tuning job."""
    table = Table(title="Fine-Tuning Job Summary")

    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Base Model", FINE_TUNE_CONFIG["model"])
    table.add_row("Training Examples", str(train_examples))
    table.add_row("Evaluation Examples", str(eval_examples))
    table.add_row("Epochs", str(FINE_TUNE_CONFIG["n_epochs"]))
    table.add_row("Job ID", job_id)

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    """Main execution flow."""
    console.print(
        Panel.fit(
            "[bold cyan]OpenAI Fine-Tuning Script[/bold cyan]\n"
            "Training a model to write like Sor Juana Inés de la Cruz",
            border_style="cyan",
        )
    )

    # Step 1: Validate API key
    console.print("\n[bold]Step 1: Validating API key[/bold]")
    client = validate_api_key()

    # Step 2: Transform data to OpenAI format
    console.print("\n[bold]Step 2: Preparing training data[/bold]")

    openai_train_file = OUTPUT_DIR / "train_openai.jsonl"
    openai_eval_file = OUTPUT_DIR / "eval_openai.jsonl"

    train_count = transform_to_openai_format(TRAIN_FILE, openai_train_file)
    eval_count = transform_to_openai_format(EVAL_FILE, openai_eval_file)

    # Step 3: Upload files
    console.print("\n[bold]Step 3: Uploading files to OpenAI[/bold]")

    train_file_id = upload_file(client, openai_train_file)
    eval_file_id = upload_file(client, openai_eval_file)

    # Step 4: Create fine-tuning job
    console.print("\n[bold]Step 4: Creating fine-tuning job[/bold]")

    job_id = create_fine_tuning_job(
        client,
        training_file_id=train_file_id,
        validation_file_id=eval_file_id,
        model=FINE_TUNE_CONFIG["model"],
        suffix="sor-juana",
    )

    # Display summary
    display_summary(train_count, eval_count, job_id)

    # Step 5: Monitor training (optional)
    console.print("[bold]Step 5: Monitor training[/bold]")
    console.print("Do you want to monitor the training progress? [Y/n]: ", end="")

    response = input().strip().lower()
    if response in ["", "y", "yes"]:
        model_id = monitor_fine_tuning_job(client, job_id)

        if model_id:
            console.print(
                Panel.fit(
                    f"[bold green]✓ Training Complete![/bold green]\n\n"
                    f"Your fine-tuned model: [cyan]{model_id}[/cyan]\n\n"
                    f"Use it with:\n"
                    f"[yellow]from openai import OpenAI\n"
                    f"client = OpenAI()\n"
                    f"response = client.chat.completions.create(\n"
                    f"    model='{model_id}',\n"
                    f"    messages=[...]\n"
                    f")[/yellow]",
                    border_style="green",
                )
            )
    else:
        console.print("\n[cyan]Job is running in the background.[/cyan]")
        console.print(f"Check status at: https://platform.openai.com/finetune/{job_id}")
        console.print("\nOr use this script to monitor:")
        console.print(f"[yellow]python scripts/monitor_job.py {job_id}[/yellow]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)
