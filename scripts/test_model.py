#!/usr/bin/env python3
"""
Test a fine-tuned model by generating text in Sor Juana's style.

Usage:
    python scripts/test_model.py <model_id>
    python scripts/test_model.py  # Uses model from fine_tuned_model.json
"""

import json
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install openai rich")
    sys.exit(1)

console = Console()

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
]


def load_model_id_from_file() -> str:
    """Load model ID from the saved training output."""
    data_dir = Path(__file__).parent.parent / "data" / "openai_training"
    model_file = data_dir / "fine_tuned_model.json"

    if not model_file.exists():
        console.print(f"[red]❌ Model file not found: {model_file}[/red]")
        console.print("\nRun the training script first or provide a model ID:")
        console.print("  python scripts/test_model.py <model_id>")
        sys.exit(1)

    with open(model_file, 'r') as f:
        data = json.load(f)

    return data.get("model_id")


def test_model(client: OpenAI, model_id: str, prompt: str, max_tokens: int = 500) -> str:
    """
    Generate text using the fine-tuned model.

    Args:
        client: OpenAI client
        model_id: Fine-tuned model ID
        prompt: User prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    system_message = (
        "Eres Sor Juana Inés de la Cruz, poeta y pensadora del siglo XVII. "
        "Escribes en estilo barroco con profundidad teológica, filosófica y feminista."
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.8,  # Higher temperature for more creative output
    )

    return response.choices[0].message.content


def compare_with_base_model(client: OpenAI, fine_tuned_model: str, prompt: str):
    """Compare fine-tuned model output with base model."""
    console.print(Panel.fit(
        f"[bold cyan]Comparison Test[/bold cyan]\n"
        f"Prompt: {prompt}",
        border_style="cyan"
    ))

    # Generate with fine-tuned model
    console.print("\n[bold green]Fine-tuned Model Output:[/bold green]")
    with console.status("[cyan]Generating with fine-tuned model...[/cyan]"):
        fine_tuned_output = test_model(client, fine_tuned_model, prompt)

    console.print(Panel(fine_tuned_output, border_style="green"))

    # Generate with base model for comparison
    console.print("\n[bold yellow]Base Model (gpt-4o-mini) Output:[/bold yellow]")
    with console.status("[cyan]Generating with base model...[/cyan]"):
        base_output = test_model(client, "gpt-4o-mini-2024-07-18", prompt)

    console.print(Panel(base_output, border_style="yellow"))


def interactive_mode(client: OpenAI, model_id: str):
    """Interactive mode for testing custom prompts."""
    console.print(Panel.fit(
        "[bold cyan]Interactive Mode[/bold cyan]\n"
        "Enter prompts to generate text in Sor Juana's style.\n"
        "Type 'quit' or 'exit' to stop.",
        border_style="cyan"
    ))

    while True:
        console.print("\n[bold]Enter your prompt:[/bold]")
        prompt = input("> ").strip()

        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        if not prompt:
            continue

        console.print()
        with console.status("[cyan]Generating...[/cyan]"):
            output = test_model(client, model_id, prompt)

        console.print(Panel(output, title="Generated Text", border_style="green"))


def main():
    console.print(Panel.fit(
        "[bold cyan]Fine-Tuned Model Testing[/bold cyan]\n"
        "Test the Sor Juana writing style model",
        border_style="cyan"
    ))

    # Get model ID
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
        console.print(f"[green]Using provided model: {model_id}[/green]")
    else:
        model_id = load_model_id_from_file()
        console.print(f"[green]Using saved model: {model_id}[/green]")

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]❌ Error: OPENAI_API_KEY environment variable not set[/red]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Show menu
    console.print("\n[bold]Choose a test mode:[/bold]")
    console.print("1. Run predefined test prompts")
    console.print("2. Compare with base model")
    console.print("3. Interactive mode (custom prompts)")
    console.print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        # Run predefined tests
        for test in TEST_PROMPTS:
            console.print(f"\n[bold cyan]{test['name']}[/bold cyan]")
            console.print(f"Prompt: [italic]{test['prompt']}[/italic]\n")

            with console.status("[cyan]Generating...[/cyan]"):
                output = test_model(client, model_id, test['prompt'])

            console.print(Panel(output, border_style="green"))
            console.print("\n" + "─" * 80 + "\n")

    elif choice == "2":
        # Compare with base model
        prompt = TEST_PROMPTS[0]['prompt']  # Use first test prompt
        compare_with_base_model(client, model_id, prompt)

    elif choice == "3":
        # Interactive mode
        interactive_mode(client, model_id)

    elif choice == "4":
        console.print("[cyan]Goodbye![/cyan]")
        return

    else:
        console.print("[red]Invalid choice[/red]")
        sys.exit(1)

    console.print("\n[green]✓ Testing complete[/green]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
