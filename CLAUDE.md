# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A DBOS-powered application for downloading, processing, and deduplicating texts from Sor Juana Inés de la Cruz across multiple sources (Project Gutenberg, Wikisource, and Biblioteca Virtual Miguel de Cervantes). The application uses durable workflow orchestration to reliably download texts concurrently, clean and normalize them, store them in DuckDB, deduplicate using MinHash LSH, and export to JSON/JSONL format.

## Environment Management

This project supports both Poetry and uv for dependency management:

**Poetry (Recommended):**
```bash
poetry install
poetry shell
```

**uv (Alternative):**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Common Commands

### CLI Usage

The application provides a CLI tool `sor-juana` with several commands:

```bash
# Build the corpus (download, process, deduplicate)
sor-juana build

# Export corpus to JSON or JSONL
sor-juana export --format jsonl --output corpus.jsonl
sor-juana export --format json -o corpus.json

# View statistics
sor-juana stats

# List texts in corpus
sor-juana list                    # All sources
sor-juana list --source gutenberg # Specific source

# Clear corpus (with confirmation)
sor-juana clear
```

### Testing
```bash
poetry run pytest
```

### Code Formatting
```bash
poetry run black .
poetry run ruff check .
```

Configuration in `pyproject.toml`:
- Line length: 120 characters
- Target: Python 3.10+

## Architecture

### Project Structure

```
sor_juana/
├── __init__.py
├── cli.py              # Click-based CLI interface
├── config.py           # Configuration and constants
├── database.py         # DuckDB operations
├── downloaders/        # Source-specific downloaders
│   ├── __init__.py
│   ├── base.py         # BaseDownloader abstract class
│   ├── gutenberg.py    # Project Gutenberg
│   ├── wikisource.py   # Wikisource
│   └── bvmc.py         # BVMC
├── processing.py       # Text cleaning and deduplication
└── workflows.py        # DBOS workflow orchestrations
```

### DBOS Workflow Orchestration

The application uses DBOS decorators to define a durable, reliable workflow (workflows.py):

- `@workflow`: Main orchestration function (`build_corpus_workflow`) that coordinates all steps
- `@step`: Individual operations (downloads, storage, deduplication) that can be retried independently
- `Queue`: Enables concurrent execution of downloads using handles and `get_result()`

**Key workflow pattern:**
1. Enqueue concurrent downloads to queue (returns handles)
2. Wait for results via `handle.get_result()`
3. Combine results
4. Store in DuckDB via `store_corpus_step`
5. Deduplicate via `deduplicate_corpus_step`
6. Export via CLI commands

### Database Architecture

**Storage:** DuckDB (database.py)
- Default location: `data/sor_juana.duckdb`
- Provides ACID transactions and efficient querying
- No external database server required

**DBOS Configuration:** Still uses PostgreSQL/SQLite for DBOS internal state
- Configured via `DBOSConfig` in config.py
- Optional: PostgreSQL via environment variable `DBOS_SYSTEM_DATABASE_URL`

**Schema (database.py:20-34):**
```sql
CREATE TABLE sor_juana_corpus (
    id INTEGER PRIMARY KEY,
    title VARCHAR,
    source VARCHAR,
    genre VARCHAR,
    url VARCHAR,
    text TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**CorpusDatabase class** provides:
- Context manager support (`with CorpusDatabase() as db:`)
- CRUD operations: `insert_text()`, `insert_many()`, `get_all_texts()`, `delete_by_id()`, `delete_many()`
- Queries: `query_by_source()`, `count()`
- Export: `export_to_json()`, `export_to_jsonl()`
- Utilities: `clear_all()`, `close()`

### Downloader Architecture

All downloaders inherit from `BaseDownloader` (downloaders/base.py) which defines:
- `download()` method: Returns `List[Dict[str, Any]]` with structure `{"text": str, "metadata": dict}`
- `source_name` property: Identifies the source

**Three downloaders:**
1. **GutenbergDownloader** (downloaders/gutenberg.py): Single URL, returns list with one item
2. **WikisourceDownloader** (downloaders/wikisource.py): MediaWiki API, limited to 50 pages
3. **BVMCDownloader** (downloaders/bvmc.py): Hardcoded list of work URLs

Each downloader uses `clean_text()` from processing.py to normalize text.

### Text Processing Pipeline

**Cleaning function** (`clean_text()`, processing.py:15-41):
- Source-specific preprocessing (removes Gutenberg headers)
- Strips HTML/XML tags using BeautifulSoup
- Normalizes whitespace
- Filters lines shorter than 20 characters
- Tokenizes into Spanish sentences using NLTK

**Deduplication** (`TextDeduplicator` class, processing.py:44-109):
- Uses MinHash LSH with threshold=0.8, num_perm=128 (configurable in config.py)
- `add_text(text_id, text)`: Computes MinHash and adds to LSH index
- `find_duplicates()`: Returns set of duplicate IDs (keeps lower ID, marks higher as duplicate)
- `deduplicate_texts(texts)`: Convenience method for batch processing

### CLI Architecture

Built with Click and Rich (cli.py):
- `main()`: Entry point, defined as `sor-juana` script in pyproject.toml
- Commands: `build`, `export`, `stats`, `list`, `clear`
- Uses Rich for formatted console output (tables, colors, status indicators)
- All commands use `CorpusDatabase` context manager
- `build` command uses DBOS workflow via `init_dbos()`, `build_corpus_workflow()`, `shutdown_dbos()`

### Configuration Management

All configuration in config.py:
- **Paths**: `PROJECT_ROOT`, `DATA_DIR`, `DB_PATH`
- **DBOS**: `DBOS_CONFIG` dict
- **Downloads**: `WIKISOURCE_PAGE_LIMIT`, `REQUEST_TIMEOUT`
- **Deduplication**: `MINHASH_THRESHOLD`, `MINHASH_NUM_PERM`

## Key Dependencies

- **click**: CLI framework
- **rich**: Terminal formatting and output
- **duckdb**: Embedded analytical database for corpus storage
- **requests**: HTTP downloads
- **beautifulsoup4 + lxml**: HTML parsing
- **mwclient**: MediaWiki API client (Wikisource)
- **dbos**: Durable workflow orchestration
- **datasketch**: MinHash LSH deduplication
- **nltk**: Spanish sentence tokenization (downloads 'punkt' on first run)
