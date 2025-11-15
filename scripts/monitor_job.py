#!/usr/bin/env python3
"""
Monitor an existing OpenAI fine-tuning job.

Usage:
    python scripts/monitor_job.py <job_id>
"""

import os
import sys
import time

try:
    from openai import OpenAI
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install openai rich")
    sys.exit(1)

console = Console()


def get_job_status(client: OpenAI, job_id: str):
    """Get and display detailed job status."""
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
    except Exception as e:
        console.print(f"[red]❌ Error retrieving job: {e}[/red]")
        sys.exit(1)

    # Create status table
    table = Table(title=f"Fine-Tuning Job: {job_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", job.status)
    table.add_row("Model", job.model)
    table.add_row("Created", str(job.created_at))

    if job.fine_tuned_model:
        table.add_row("Fine-tuned Model", job.fine_tuned_model)

    if job.finished_at:
        table.add_row("Finished", str(job.finished_at))

    if job.error:
        table.add_row("Error", str(job.error))

    # Add training metrics if available
    if hasattr(job, "result_files") and job.result_files:
        table.add_row("Result Files", str(len(job.result_files)))

    console.print(table)

    # Show recent events if available
    try:
        events = client.fine_tuning.jobs.list_events(job_id, limit=10)
        if events.data:
            console.print("\n[bold cyan]Recent Events:[/bold cyan]")
            for event in events.data:
                console.print(f"  [{event.created_at}] {event.message}")
    except Exception:
        pass

    return job


def monitor_until_complete(client: OpenAI, job_id: str, poll_interval: int = 60):
    """Monitor job until it completes."""
    console.print(f"\n[cyan]Monitoring job {job_id}...[/cyan]")
    console.print(f"Checking every {poll_interval} seconds (Ctrl+C to stop)\n")

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

                progress.update(task, description=f"Status: {status}")

                if status == "succeeded":
                    progress.stop()
                    console.print("\n[green]✓ Fine-tuning completed successfully![/green]")
                    console.print(f"\n[bold green]Fine-tuned model:[/bold green] {job.fine_tuned_model}")

                    console.print(
                        Panel.fit(
                            f"[bold green]Training Complete![/bold green]\n\n"
                            f"Model ID: [cyan]{job.fine_tuned_model}[/cyan]\n\n"
                            f"Use it in your code:\n"
                            f"[yellow]from openai import OpenAI\n"
                            f"client = OpenAI()\n"
                            f"response = client.chat.completions.create(\n"
                            f"    model='{job.fine_tuned_model}',\n"
                            f"    messages=[...]\n"
                            f")[/yellow]",
                            border_style="green",
                        )
                    )
                    return

                elif status in ["failed", "cancelled"]:
                    progress.stop()
                    console.print(f"\n[red]❌ Job {status}[/red]")
                    if job.error:
                        console.print(f"[red]Error: {job.error}[/red]")
                    return

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            progress.stop()
            console.print("\n[yellow]Monitoring stopped. Job is still running.[/yellow]")


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python scripts/monitor_job.py <job_id>[/red]")
        sys.exit(1)

    job_id = sys.argv[1]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]❌ Error: OPENAI_API_KEY environment variable not set[/red]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Show current status
    job = get_job_status(client, job_id)

    # Ask if user wants continuous monitoring
    if job.status not in ["succeeded", "failed", "cancelled"]:
        console.print("\n[cyan]Job is still running.[/cyan]")
        console.print("Monitor until completion? [Y/n]: ", end="")

        response = input().strip().lower()
        if response in ["", "y", "yes"]:
            monitor_until_complete(client, job_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        sys.exit(1)
