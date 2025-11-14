"""CLI interface for Sor Juana Downloader."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from .workflows import build_corpus_workflow, export_csv_workflow, launch_dbos, shutdown_dbos
from .database import CorpusDatabase
from .config import DATA_DIR

console = Console()


@click.group()
@click.version_option(version="0.1.0")
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


if __name__ == "__main__":
    main()
