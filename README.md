# Sor Juana Downloader

A DBOS-powered application for downloading, processing, and deduplicating texts from Sor Juana Inés de la Cruz across multiple sources.

## Features

- **Multi-source downloads**: Project Gutenberg, Wikisource, and Biblioteca Virtual Miguel de Cervantes
- **Intelligent text cleaning**: Removes headers, HTML tags, and normalizes text
- **Deduplication**: Uses MinHash LSH for efficient similarity detection
- **DBOS orchestration**: Reliable, durable workflow execution with concurrent downloads
- **DuckDB storage**: Embedded analytical database for efficient corpus storage and querying
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

# Run the application
python sor_juana_downloader.py
```

#### Option 2: Using uv

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies from pyproject.toml
uv pip install -e .

# Run the application
python sor_juana_downloader.py
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
2. Clean and normalize the text
3. Store texts in DuckDB
4. Deduplicate using MinHash LSH

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
```

### List texts

```bash
# List all texts
sor-juana list

# List texts from specific source
sor-juana list --source gutenberg
sor-juana list --source wikisource
sor-juana list --source bvmc
```

### Clear corpus

```bash
sor-juana clear
```

## Output

The corpus is stored in DuckDB (`data/sor_juana.duckdb`) and can be exported to JSON or JSONL format:

**JSONL format:**
```json
{"text": "...", "metadata": {"source": "gutenberg", "title": "Obras selectas", ...}}
{"text": "...", "metadata": {"source": "wikisource", "title": "...", ...}}
```

**JSON format:**
```json
[
  {"text": "...", "metadata": {"source": "gutenberg", "title": "Obras selectas", ...}},
  {"text": "...", "metadata": {"source": "wikisource", "title": "...", ...}}
]
```

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
│   └── workflows.py          # DBOS workflows
├── data/                     # Generated data directory
│   └── sor_juana.duckdb      # DuckDB database
├── pyproject.toml            # Poetry configuration
├── .python-version           # Python version for uv
├── README.md                 # This file
└── CLAUDE.md                 # Claude Code guidance
```

## Dependencies

- **click**: CLI framework
- **rich**: Terminal formatting and output
- **duckdb**: Embedded analytical database
- **requests**: HTTP client for downloading texts
- **beautifulsoup4**: HTML parsing and cleaning
- **mwclient**: MediaWiki API client for Wikisource
- **dbos**: Durable workflow orchestration
- **datasketch**: MinHash LSH for deduplication
- **nltk**: Natural language processing (sentence tokenization)

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

