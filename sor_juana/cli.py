"""CLI interface for Sor Juana Downloader."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from .workflows import build_corpus_workflow, export_csv_workflow, launch_dbos, shutdown_dbos
from .database import CorpusDatabase
from .config import DATA_DIR

console = Console()


@click.group()
@click.version_option(version="0.2.0")
def main():
    """Sor Juana Downloader - Build and manage a corpus of Sor Juana Inés de la Cruz texts."""
    pass


@main.command()
def build():
    """Download, process, and deduplicate the Sor Juana corpus."""
    console.print("\n[bold cyan]Starting corpus build...[/bold cyan]\n")

    try:
        # Launch DBOS
        launch_dbos()

        # Run workflow
        with console.status("[bold green]Running DBOS workflow..."):
            result = build_corpus_workflow()

        # Display results
        console.print("[bold green]✓[/bold green] Corpus build completed!\n")
        console.print(f"  Downloaded: [cyan]{result['total_downloaded']}[/cyan] texts")
        console.print(f"  Stored: [cyan]{result['total_stored']}[/cyan] texts")
        console.print(f"  Duplicates removed: [yellow]{result['duplicates_removed']}[/yellow]")
        console.print(f"  Final count: [bold green]{result['final_count']}[/bold green] texts\n")

        # Shutdown DBOS
        shutdown_dbos()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@main.command()
@click.option(
    "--format",
    type=click.Choice(["json", "jsonl"], case_sensitive=False),
    default="jsonl",
    help="Output format (default: jsonl)",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: data/corpus.<format>)")
def export(format: str, output: str):
    """Export the corpus to JSON or JSONL format."""
    try:
        with CorpusDatabase() as db:
            # Determine output path
            if output:
                output_path = Path(output)
            else:
                output_path = DATA_DIR / f"corpus.{format}"

            # Export based on format
            with console.status(f"[bold green]Exporting to {format.upper()}..."):
                if format == "json":
                    count = db.export_to_json(output_path)
                else:
                    count = db.export_to_jsonl(output_path)

            console.print(f"\n[bold green]✓[/bold green] Exported {count} texts to [cyan]{output_path}[/cyan]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@main.command()
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: data/corpus.csv)")
def export_csv(output: str):
    """Export the corpus to CSV format using DBOS workflow."""
    console.print("\n[bold cyan]Starting CSV export...[/bold cyan]\n")

    try:
        # Launch DBOS
        launch_dbos()

        # Determine output path
        if output:
            output_path = Path(output)
        else:
            output_path = DATA_DIR / "corpus.csv"

        # Run workflow
        with console.status("[bold green]Running DBOS export workflow..."):
            result = export_csv_workflow(str(output_path))

        # Display results
        console.print("[bold green]✓[/bold green] CSV export completed!\n")
        console.print(f"  Output: [cyan]{result['output_path']}[/cyan]")
        console.print(f"  Exported: [cyan]{result['count']}[/cyan] texts\n")

        # Shutdown DBOS
        shutdown_dbos()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@main.command()
def stats():
    """Display corpus statistics."""
    try:
        with CorpusDatabase() as db:
            total = db.count()

            if total == 0:
                console.print("\n[yellow]Corpus is empty. Run 'sor-juana build' first.[/yellow]\n")
                return

            # Create statistics table
            table = Table(title="Corpus Statistics", show_header=True, header_style="bold cyan")
            table.add_column("Source", style="cyan")
            table.add_column("Count", justify="right", style="green")

            # Query by source
            sources = ["gutenberg", "wikisource", "bvmc"]
            for source in sources:
                texts = db.query_by_source(source)
                table.add_row(source.title(), str(len(texts)))

            table.add_section()
            table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")

            console.print()
            console.print(table)
            console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@main.command()
@click.option("--source", type=click.Choice(["gutenberg", "wikisource", "bvmc", "all"]), default="all")
def list(source: str):
    """List texts in the corpus."""
    try:
        with CorpusDatabase() as db:
            # Get texts
            if source == "all":
                texts = db.get_all_texts()
                title_suffix = "All Sources"
            else:
                texts = db.query_by_source(source)
                title_suffix = source.title()

            if not texts:
                console.print(f"\n[yellow]No texts found for {title_suffix}.[/yellow]\n")
                return

            # Create table
            table = Table(title=f"Corpus Texts - {title_suffix}", show_header=True, header_style="bold cyan")
            table.add_column("ID", justify="right", style="dim")
            table.add_column("Title", style="cyan")
            table.add_column("Source", style="green")
            table.add_column("Length", justify="right")

            for text in texts[:50]:  # Limit to first 50
                metadata = text["metadata"]
                table.add_row(
                    str(text["id"]), metadata.get("title", "N/A"), metadata.get("source", "N/A"), str(len(text["text"]))
                )

            console.print()
            console.print(table)
            if len(texts) > 50:
                console.print(f"\n[dim]Showing first 50 of {len(texts)} texts[/dim]\n")
            else:
                console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@main.command()
@click.option(
    "--eval-ratio",
    type=float,
    default=0.15,
    help="Proportion of data for eval set (default: 0.15 = 15%%)",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for train.jsonl and eval.jsonl (default: data/)",
)
def split(eval_ratio: float, seed: int, output_dir: str):
    """Split corpus into train and eval sets and save as JSONL files."""
    console.print("\n[bold cyan]Splitting corpus into train/eval sets...[/bold cyan]\n")

    try:
        with CorpusDatabase() as db:
            total = db.count()

            if total == 0:
                console.print("\n[yellow]Corpus is empty. Run 'sor-juana build' first.[/yellow]\n")
                return

            # Validate eval_ratio
            if not 0.1 <= eval_ratio <= 0.2:
                console.print(
                    f"\n[yellow]Warning: eval_ratio {eval_ratio:.1%} is outside recommended 10-20% range.[/yellow]\n"
                )

            # Determine output directory
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = DATA_DIR

            output_path.mkdir(parents=True, exist_ok=True)

            # Perform split
            with console.status("[bold green]Splitting corpus..."):
                counts = db.split_train_eval(eval_ratio=eval_ratio, seed=seed)

            console.print("[bold green]✓[/bold green] Split completed!\n")
            console.print(f"  Total texts: [cyan]{total}[/cyan]")
            console.print(f"  Train set: [green]{counts['train']}[/green] ({counts['train']/total:.1%})")
            console.print(f"  Eval set: [yellow]{counts['eval']}[/yellow] ({counts['eval']/total:.1%})")
            console.print(f"  Random seed: [dim]{seed}[/dim]\n")

            # Export train set
            train_path = output_path / "train.jsonl"
            with console.status(f"[bold green]Exporting train set to {train_path}..."):
                train_count = db.export_split_to_jsonl("train", train_path)

            # Export eval set
            eval_path = output_path / "eval.jsonl"
            with console.status(f"[bold green]Exporting eval set to {eval_path}..."):
                eval_count = db.export_split_to_jsonl("eval", eval_path)

            console.print("\n[bold green]✓[/bold green] Export completed!\n")
            console.print(f"  Train: [cyan]{train_path}[/cyan] ({train_count} texts)")
            console.print(f"  Eval: [cyan]{eval_path}[/cyan] ({eval_count} texts)\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear the entire corpus?")
def clear():
    """Clear all texts from the corpus (requires confirmation)."""
    try:
        with CorpusDatabase() as db:
            count = db.count()
            db.clear_all()
            console.print(f"\n[bold green]✓[/bold green] Cleared {count} texts from corpus\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


# Training and Evaluation Commands

@main.group()
def train():
    """Fine-tuning and training commands."""
    pass


@train.command(name="openai")
@click.option("--train-file", type=click.Path(exists=True), help="Training JSONL file (default: data/train.jsonl)")
@click.option("--eval-file", type=click.Path(exists=True), help="Evaluation JSONL file (default: data/eval.jsonl)")
@click.option("--model", default="gpt-4o-mini-2024-07-18", help="Base model to fine-tune")
@click.option("--suffix", default="sor-juana", help="Suffix for fine-tuned model name")
@click.option("--monitor/--no-monitor", default=True, help="Monitor training progress")
def train_openai(train_file: str, eval_file: str, model: str, suffix: str, monitor: bool):
    """Start OpenAI fine-tuning job using DBOS workflows."""
    from .training import full_fine_tuning_workflow
    
    console.print("\n[bold cyan]Starting OpenAI Fine-Tuning...[/bold cyan]\n")
    
    try:
        # Launch DBOS
        launch_dbos()
        
        # Determine file paths
        train_path = train_file or str(DATA_DIR / "train.jsonl")
        eval_path = eval_file or str(DATA_DIR / "eval.jsonl")
        
        # Check files exist
        if not Path(train_path).exists():
            console.print(f"[red]✗ Training file not found: {train_path}[/red]")
            console.print("Run 'sor-juana split' first to create training data.")
            raise click.Abort()
            
        if not Path(eval_path).exists():
            console.print(f"[red]✗ Evaluation file not found: {eval_path}[/red]")
            console.print("Run 'sor-juana split' first to create evaluation data.")
            raise click.Abort()
        
        # Run workflow
        with console.status("[bold green]Running fine-tuning workflow..."):
            result = full_fine_tuning_workflow(
                train_jsonl=train_path,
                eval_jsonl=eval_path,
                monitor=monitor,
            )
        
        # Display results
        console.print("\n[bold green]✓[/bold green] Fine-tuning workflow completed!\n")
        
        prep = result["preparation"]
        job = result["job"]
        
        console.print(f"  Train examples: [cyan]{prep['train']['examples_processed']}[/cyan]")
        console.print(f"  Eval examples: [cyan]{prep['eval']['examples_processed']}[/cyan]")
        console.print(f"  Job ID: [yellow]{job['job_id']}[/yellow]")
        console.print(f"  Status: [cyan]{job['status']}[/cyan]")
        
        if monitor and "monitor" in result:
            monitor_result = result["monitor"]
            if monitor_result["status"] == "succeeded":
                console.print(f"\n  [bold green]Fine-tuned model:[/bold green] [cyan]{monitor_result['fine_tuned_model']}[/cyan]")
            else:
                console.print(f"\n  Final status: [yellow]{monitor_result['status']}[/yellow]")
        else:
            console.print(f"\n  Check status with: [yellow]sor-juana train monitor {job['job_id']}[/yellow]")
        
        console.print()
        
        # Shutdown DBOS
        shutdown_dbos()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@train.command(name="monitor")
@click.argument("job_id")
@click.option("--poll-interval", default=60, help="Seconds between status checks")
def train_monitor(job_id: str, poll_interval: int):
    """Monitor an OpenAI fine-tuning job."""
    from .training import monitor_fine_tuning_workflow, get_job_info_workflow
    
    console.print(f"\n[bold cyan]Monitoring Fine-Tuning Job: {job_id}[/bold cyan]\n")
    
    try:
        # Launch DBOS
        launch_dbos()
        
        # Get initial status
        with console.status("[bold green]Getting job info..."):
            info = get_job_info_workflow(job_id)
        
        # Display status
        status = info["status"]
        table = Table(title=f"Job Status: {job_id}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", status["status"])
        table.add_row("Model", status["model"])
        table.add_row("Created", str(status.get("created_at", "N/A")))
        
        if status.get("fine_tuned_model"):
            table.add_row("Fine-tuned Model", status["fine_tuned_model"])
            
        if status.get("finished_at"):
            table.add_row("Finished", str(status["finished_at"]))
            
        console.print(table)
        
        # Show recent events
        if info["recent_events"]:
            console.print("\n[bold cyan]Recent Events:[/bold cyan]")
            for event in info["recent_events"]:
                console.print(f"  [{event['created_at']}] {event['message']}")
        
        # Monitor if not complete
        if status["status"] not in ["succeeded", "failed", "cancelled"]:
            console.print(f"\n[cyan]Monitoring until completion (checking every {poll_interval}s)...[/cyan]")
            console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")
            
            result = monitor_fine_tuning_workflow(job_id, poll_interval)
            
            if result["status"] == "succeeded":
                console.print(Panel.fit(
                    f"[bold green]Training Complete![/bold green]\n\n"
                    f"Model ID: [cyan]{result['fine_tuned_model']}[/cyan]\n\n"
                    f"Test it with:\n"
                    f"[yellow]sor-juana test model {result['fine_tuned_model']}[/yellow]",
                    border_style="green",
                ))
            else:
                console.print(f"\n[yellow]Job finished with status: {result['status']}[/yellow]\n")
        
        # Shutdown DBOS
        shutdown_dbos()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped. Job is still running.[/yellow]")
        shutdown_dbos()
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@train.command(name="local")
@click.option("--csv", type=click.Path(exists=True), help="CSV file with prompt/completion pairs")
@click.option("--model", default="meta-llama/Llama-2-7b", help="Base model name")
@click.option("--output", default="./sor_juana_model", help="Output directory")
@click.option("--eval-only", is_flag=True, help="Run evaluation only (no fine-tuning)")
@click.option("--sample-size", default=20, help="Number of examples to evaluate")
def train_local(csv: str, model: str, output: str, eval_only: bool, sample_size: int):
    """Local MLX-based training and evaluation (Apple Silicon)."""
    from .local_training import local_training_pipeline_workflow
    
    console.print("\n[bold cyan]Local Training Pipeline (Apple Silicon / MLX)[/bold cyan]\n")
    
    try:
        # Determine CSV path
        csv_path = csv or str(DATA_DIR / "corpus.csv")
        
        if not Path(csv_path).exists():
            console.print(f"[red]✗ CSV file not found: {csv_path}[/red]")
            console.print("Run 'sor-juana export-csv' first to create CSV file.")
            raise click.Abort()
        
        # Launch DBOS
        launch_dbos()
        
        # Run workflow
        with console.status("[bold green]Running local training pipeline..."):
            result = local_training_pipeline_workflow(
                csv_path=csv_path,
                model_name=model,
                output_dir=output,
                eval_only=eval_only,
                sample_size=sample_size,
            )
        
        # Display results
        if result["status"] == "error":
            console.print(f"\n[red]✗ Pipeline failed:[/red] {result['error']}\n")
        else:
            console.print("\n[bold green]✓[/bold green] Pipeline completed!\n")
            console.print(f"  Model: [cyan]{result['model_name']}[/cyan]")
            console.print(f"  Train examples: [cyan]{result['dataset']['train_count']}[/cyan]")
            console.print(f"  Eval examples: [cyan]{result['dataset']['eval_count']}[/cyan]")
            
            eval_result = result["evaluation"]
            console.print(f"\n  [bold]Evaluation Results:[/bold]")
            console.print(f"    Pass rate: [green]{eval_result['pass_rate']:.1%}[/green]")
            console.print(f"    Mean score: [green]{eval_result['mean_score']:.2f}/5.0[/green]")
            console.print(f"    Evaluated: [cyan]{eval_result['total_evaluated']}[/cyan] examples")
            console.print(f"    Results: [cyan]{eval_result['results_file']}[/cyan]")
        
        console.print()
        
        # Shutdown DBOS
        shutdown_dbos()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


@main.group()
def test():
    """Model testing and evaluation commands."""
    pass


@test.command(name="model")
@click.argument("model_id", required=False)
@click.option("--prompt", "-p", help="Single prompt to test")
@click.option("--compare", is_flag=True, help="Compare with base model")
def test_model(model_id: str, prompt: str, compare: bool):
    """Test a fine-tuned model with prompts."""
    from .evaluation import test_model_with_prompt_workflow, test_model_with_prompts_workflow, compare_models_workflow
    
    console.print("\n[bold cyan]Model Testing[/bold cyan]\n")
    
    try:
        # Launch DBOS
        launch_dbos()
        
        if prompt:
            # Test single prompt
            with console.status("[bold green]Generating response..."):
                if compare:
                    result = compare_models_workflow(prompt, model_id)
                    
                    # Display comparison
                    console.print(Panel.fit(
                        f"[bold cyan]Prompt:[/bold cyan]\n{prompt}",
                        border_style="cyan"
                    ))
                    
                    console.print("\n[bold green]Fine-tuned Model:[/bold green]")
                    console.print(Panel(result["fine_tuned"]["generated_text"], border_style="green"))
                    ft_eval = result["fine_tuned"]["evaluation"]
                    console.print(f"  Score: [green]{ft_eval['overall_score']:.2f}/5.0[/green]")
                    
                    console.print("\n[bold yellow]Base Model:[/bold yellow]")
                    console.print(Panel(result["base"]["generated_text"], border_style="yellow"))
                    base_eval = result["base"]["evaluation"]
                    console.print(f"  Score: [yellow]{base_eval['overall_score']:.2f}/5.0[/yellow]")
                    
                    diff = result["score_difference"]
                    color = "green" if diff > 0 else "red"
                    console.print(f"\n  Score difference: [{color}]{diff:+.2f}[/{color}]")
                else:
                    result = test_model_with_prompt_workflow(prompt, model_id)
                    
                    console.print(Panel.fit(
                        f"[bold cyan]Prompt:[/bold cyan]\n{prompt}",
                        border_style="cyan"
                    ))
                    
                    console.print("\n[bold green]Generated:[/bold green]")
                    console.print(Panel(result["generated_text"], border_style="green"))
                    
                    if "evaluation" in result:
                        eval_result = result["evaluation"]
                        console.print(f"\n  Overall score: [green]{eval_result['overall_score']:.2f}/5.0[/green]")
                        console.print(f"  Baroque style: {eval_result['baroque_style']:.2f}")
                        console.print(f"  Thematic alignment: {eval_result['thematic_alignment']:.2f}")
        else:
            # Test with multiple prompts
            with console.status("[bold green]Running test suite..."):
                result = test_model_with_prompts_workflow(None, model_id)
            
            console.print(f"[bold green]✓[/bold green] Test suite completed!\n")
            console.print(f"  Model: [cyan]{result['model_id']}[/cyan]")
            console.print(f"  Total tests: [cyan]{result['total_tests']}[/cyan]")
            console.print(f"  Passed: [green]{result['passed']}[/green] ({result['pass_rate']:.1%})")
            console.print(f"  Mean score: [green]{result['mean_score']:.2f}/5.0[/green]\n")
            
            # Show details for each test
            for test_result in result["results"]:
                name = test_result["name"]
                eval_result = test_result.get("evaluation", {})
                score = eval_result.get("overall_score", 0)
                passed = "✓" if eval_result.get("passes_threshold", False) else "✗"
                
                color = "green" if eval_result.get("passes_threshold", False) else "yellow"
                console.print(f"  [{color}]{passed}[/{color}] {name}: {score:.2f}/5.0")
        
        console.print()
        
        # Shutdown DBOS
        shutdown_dbos()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        import traceback
        console.print(traceback.format_exc())
        raise click.Abort()


@test.command(name="sample")
@click.option("--prompt", "-p", required=True, help="Prompt to test")
@click.option("--model", default="meta-llama/Llama-2-7b", help="Model name")
def test_sample(prompt: str, model: str):
    """Generate and evaluate a sample with local model."""
    from .local_training import generate_sample_workflow
    
    console.print("\n[bold cyan]Sample Generation[/bold cyan]\n")
    
    try:
        # Launch DBOS
        launch_dbos()
        
        # Run workflow
        with console.status("[bold green]Generating sample..."):
            result = generate_sample_workflow(prompt, model)
        
        # Display results
        console.print(Panel.fit(
            f"[bold cyan]Prompt:[/bold cyan]\n{prompt}",
            border_style="cyan"
        ))
        
        console.print("\n[bold green]Generated:[/bold green]")
        console.print(Panel(result["generated"], border_style="green"))
        
        eval_result = result["evaluation"]
        console.print(f"\n  Overall score: [green]{eval_result['overall_score']:.2f}/5.0[/green]")
        console.print(f"  Baroque style: {eval_result['baroque_style']:.2f}")
        console.print(f"  Thematic alignment: {eval_result['thematic_alignment']:.2f}")
        console.print(f"  Linguistic authenticity: {eval_result['linguistic_authenticity']:.2f}")
        console.print(f"  Structural coherence: {eval_result['structural_coherence']:.2f}")
        
        console.print()
        
        # Shutdown DBOS
        shutdown_dbos()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise click.Abort()


if __name__ == "__main__":
    main()
