# Sor Juana Downloader

A DBOS-powered application for downloading, processing, and deduplicating texts from Sor Juana Inés de la Cruz across multiple sources.

## Features

- **Multi-source downloads**: Project Gutenberg, Wikisource, and Biblioteca Virtual Miguel de Cervantes
- **Semantic HTML chunking**: Preserves document structure (headings, paragraphs, poetry) with rich metadata
- **Intelligent text cleaning**: Removes headers, HTML tags, and normalizes text
- **Deduplication**: Uses MinHash LSH for efficient similarity detection
- **DBOS orchestration**: Reliable, durable workflow execution with concurrent downloads
- **DuckDB storage**: Embedded analytical database for efficient corpus storage and querying
- **Multiple export formats**: JSON, JSONL, and CSV export options
- **Train/Eval splitting**: Random split with reproducible seeds for ML training datasets
- **CLI interface**: User-friendly commands for building, exporting, and managing the corpus

## Setup

This project uses both `uv` and `poetry` for Python environment management.

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- [Poetry](https://python-poetry.org/) - Python dependency management

### Installation

#### Option 1: Using Poetry (Recommended)

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

#### Option 2: Using uv

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies from pyproject.toml
uv pip install -e .
```

#### Option 3: Using both (for development)

```bash
# Use uv for fast installation with poetry
uv pip install poetry
poetry install
```

## Configuration

### Database Configuration

By default, the application uses SQLite. To use PostgreSQL:

```bash
export DBOS_SYSTEM_DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
```

### Environment Variables

- `DBOS_SYSTEM_DATABASE_URL`: Database connection string (optional, defaults to SQLite)

## Usage

The application provides a CLI tool for building and managing the corpus:

### Build the corpus

```bash
sor-juana build
```

This will:
1. Download texts from Project Gutenberg, Wikisource, and BVMC concurrently
2. Parse HTML content semantically to preserve document structure
3. Extract chunks with metadata (headings, paragraphs, poetry, page numbers)
4. Clean and normalize the text
5. Store texts in DuckDB
6. Deduplicate using MinHash LSH

### View statistics

```bash
sor-juana stats
```

### Export corpus

```bash
# Export to JSONL (default)
sor-juana export --format jsonl --output corpus.jsonl

# Export to JSON
sor-juana export --format json -o corpus.json

# Export to CSV using DuckDB native export
sor-juana export-csv --output corpus.csv
```

### Split into train/eval sets

Split the corpus randomly into training and evaluation sets for ML model training:

```bash
# Default split (15% eval, seed=42)
sor-juana split

# Custom eval ratio (20% eval)
sor-juana split --eval-ratio 0.2

# Custom seed and output directory
sor-juana split --seed 123 --output-dir ./splits
```

This creates `train.jsonl` and `eval.jsonl` files in the specified directory (default: `data/`).

### List texts

```bash
# List all texts
sor-juana list
# or explicitly
sor-juana list --source all

# List texts from specific source
sor-juana list --source gutenberg
sor-juana list --source wikisource
sor-juana list --source bvmc
```

### Clear corpus

```bash
sor-juana clear
```

## Model Training & Evaluation

The application includes DBOS-powered workflows for fine-tuning and evaluating language models on Sor Juana's writing style.

### OpenAI Fine-tuning

Train a GPT-4o-mini model using OpenAI's fine-tuning API with durable DBOS workflows:

```bash
# Start fine-tuning with automatic monitoring
sor-juana train openai

# Start without monitoring (check status later)
sor-juana train openai --no-monitor

# Monitor an existing job
sor-juana train monitor <job_id>

# Custom training files
sor-juana train openai --train-file path/to/train.jsonl --eval-file path/to/eval.jsonl
```

**Features:**
- ✅ Durable execution: Automatically resumes if interrupted
- ✅ Automatic file upload and job creation
- ✅ Real-time monitoring with status updates
- ✅ Saves model ID to `data/openai_training/fine_tuned_model.json`

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Run `sor-juana split` first to generate train/eval sets

**Configuration:**
- Base model: `gpt-4o-mini-2024-07-18` (configurable)
- Default epochs: 3
- Batch size: Auto
- Learning rate multiplier: Auto

### Model Testing

Test your fine-tuned models with comprehensive evaluation:

```bash
# Test with predefined prompts (uses saved model ID)
sor-juana test model

# Test with specific model ID
sor-juana test model ft:gpt-4o-mini-2024-07-18:org:model-id:suffix

# Test with custom prompt
sor-juana test model --prompt "Escribe un soneto sobre el conocimiento"

# Compare fine-tuned vs base model
sor-juana test model --prompt "Your prompt" --compare
```

**Evaluation Metrics:**
- **Baroque style** (0-1): Rhetorical features, complex sentences
- **Thematic alignment** (0-1): Feminist/intellectual/theological themes
- **Linguistic authenticity** (0-1): Period-appropriate vocabulary
- **Structural coherence** (0-1): Sonnet/décima structure, rhyme patterns
- **Overall score** (1-5): Weighted composite score

### Local Training (Apple Silicon)

For Apple Silicon Macs (M1/M2/M3/M4), use MLX for local training:

```bash
# Run local training pipeline
sor-juana train local --csv corpus.csv

# Evaluation only (no fine-tuning)
sor-juana train local --eval-only

# Generate and evaluate sample
sor-juana test sample --prompt "Escribe un soneto sobre el conocimiento" --model meta-llama/Llama-2-7b
```

**Features:**
- MLX-optimized workflows for Apple Silicon
- Comprehensive style evaluation
- Batch evaluation on held-out data
- Sample generation with authenticity scoring

**Requirements:**
- Apple Silicon Mac (M1/M2/M3/M4)
- Install: `pip install mlx mlx-lm transformers torch`
- CSV file with `prompt` and `completion` columns (create with `sor-juana export-csv`)

### Workflow Features

All training and evaluation operations use DBOS workflows with:
- **Durable execution**: Automatically resume from last completed step if interrupted
- **Reliability**: Built-in retry logic and error handling
- **Observability**: Track progress via DBOS admin server (port 3001)
- **Concurrency**: Managed parallel execution with queues

See [`scripts/README.md`](scripts/README.md) for complete CLI reference and advanced usage.

## Output

The corpus is stored in DuckDB (`data/sor_juana.duckdb`) and can be exported to JSON, JSONL, or CSV format.

### Data Structure

Each text chunk includes rich metadata:

**JSONL/JSON format:**
```json
{
  "text": "El sexo llamado débil y mirado con desden...",
  "metadata": {
    "source": "gutenberg",
    "title": "Obras selectas",
    "url": "https://www.gutenberg.org/...",
    "genre": "poetry_prose",
    "chunk_number": 13,
    "chunk_type": "paragraph",
    "section_title": "BIOGRAFIA",
    "section_hierarchy": ["BIOGRAFIA", "I"],
    "page_number": null
  }
}
```

**CSV format:**
The CSV export includes columns: `id`, `title`, `source`, `genre`, `url`, `page_number`, `chunk_number`, `chunk_type`, `text`, and `metadata` (as JSON string).

### Train/Eval Splits

After running `sor-juana split`, you'll have:
- `data/train.jsonl` - Training set (85% by default)
- `data/eval.jsonl` - Evaluation set (15% by default)

Both files follow the same JSONL format as above.

## Development

### Running tests

```bash
poetry run pytest
```

### Code formatting

```bash
poetry run black .
poetry run ruff check .
```

## Project Structure

```
sor-night-ws/
├── sor_juana/                # Main package
│   ├── __init__.py
│   ├── cli.py                # CLI interface
│   ├── config.py             # Configuration
│   ├── database.py           # DuckDB operations
│   ├── downloaders/          # Source-specific downloaders
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gutenberg.py
│   │   ├── wikisource.py
│   │   └── bvmc.py
│   ├── processing.py         # Text cleaning and deduplication
│   ├── workflows.py          # Corpus building DBOS workflows
│   ├── training.py           # OpenAI fine-tuning DBOS workflows
│   ├── evaluation.py         # Model testing DBOS workflows
│   └── local_training.py     # Local MLX training DBOS workflows
├── data/                     # Generated data directory
│   ├── sor_juana.duckdb      # DuckDB database
│   ├── train.jsonl          # Training set (after split)
│   ├── eval.jsonl           # Evaluation set (after split)
│   └── openai_training/      # OpenAI training outputs
│       └── fine_tuned_model.json
├── scripts/                 # Utility scripts
│   ├── quick_start.sh       # Quick setup script
│   └── README.md            # Scripts documentation
├── pyproject.toml            # Poetry configuration
├── .python-version           # Python version for uv
├── README.md                 # This file
├── CHANGELOG.md             # Version history
└── CLAUDE.md                 # Claude Code guidance
```

## Dependencies

### Core Dependencies
- **click**: CLI framework
- **rich**: Terminal formatting and output
- **duckdb**: Embedded analytical database
- **requests**: HTTP client for downloading texts
- **beautifulsoup4**: HTML parsing and cleaning
- **mwclient**: MediaWiki API client for Wikisource
- **dbos**: Durable workflow orchestration
- **datasketch**: MinHash LSH for deduplication
- **nltk**: Natural language processing (sentence tokenization)

### Optional Dependencies
- **openai**: OpenAI API client (for fine-tuning)
- **mlx**: MLX framework for Apple Silicon (for local training)
- **mlx-lm**: MLX language models (for local training)
- **transformers**: HuggingFace transformers (for local training)
- **torch**: PyTorch (for local training fallback)

See [`pyproject.toml`](pyproject.toml) for complete dependency list.

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and release notes.

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

For major changes, please:
1. Update the CHANGELOG.md
2. Update version numbers in `pyproject.toml` and `sor_juana/cli.py`
3. Follow semantic versioning (https://semver.org/)

