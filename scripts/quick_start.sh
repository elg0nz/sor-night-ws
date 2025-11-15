#!/bin/bash
# Quick start script for OpenAI fine-tuning
# This script guides you through the complete process

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Sor Juana Fine-Tuning Quick Start                        ║"
echo "║   Teaching an LLM to write like Sor Juana Inés de la Cruz ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY is not set"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    read -p "Press Enter to continue after setting the key, or Ctrl+C to exit..."

    # Check again
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ API key still not set. Exiting."
        exit 1
    fi
fi

echo "✓ OpenAI API key found"
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python3 -c "import openai, rich" 2>/dev/null; then
    echo "⚠️  Missing dependencies. Installing..."
    pip install openai rich
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi
echo ""

# Check if corpus exists
if [ ! -f "data/train.jsonl" ] || [ ! -f "data/eval.jsonl" ]; then
    echo "⚠️  Training data not found"
    echo ""
    echo "Building corpus first..."
    sor-juana build
    echo "✓ Corpus built"
else
    echo "✓ Training data found"
fi
echo ""

# Show what will happen
echo "═══════════════════════════════════════════════════════════════"
echo "This script will:"
echo "  1. Transform corpus data to OpenAI format"
echo "  2. Upload files to OpenAI"
echo "  3. Start a fine-tuning job"
echo "  4. Monitor training progress"
echo ""
echo "Estimated time: 20-60 minutes"
echo "Estimated cost: \$8-15 USD"
echo "═══════════════════════════════════════════════════════════════"
echo ""

read -p "Continue with fine-tuning? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting fine-tuning process..."
echo ""

# Run the training script
python3 scripts/train_openai.py

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     Quick Start Complete!                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  • Test your model: python scripts/test_model.py"
echo "  • Monitor jobs: python scripts/monitor_job.py <job_id>"
echo "  • Check the scripts/README.md for more options"
echo ""
