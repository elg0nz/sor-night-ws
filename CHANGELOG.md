# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-12-XX

### Added

#### DBOS Workflows for Training & Evaluation
- **`sor_juana/training.py`**: Complete OpenAI fine-tuning workflows
  - `prepare_training_data_workflow`: Transform corpus to OpenAI format
  - `start_fine_tuning_workflow`: Upload files and create fine-tuning jobs
  - `monitor_fine_tuning_workflow`: Durable monitoring with automatic recovery
  - `full_fine_tuning_workflow`: End-to-end training pipeline
  - `get_job_info_workflow`: Retrieve job status and events

- **`sor_juana/evaluation.py`**: Model testing and evaluation workflows
  - `test_model_with_prompt_workflow`: Single prompt testing with evaluation
  - `test_model_with_prompts_workflow`: Batch testing with predefined prompts
  - `compare_models_workflow`: Compare fine-tuned vs base model outputs
  - `save_test_results_workflow`: Persist evaluation results
  - Comprehensive style scoring (Baroque, thematic, linguistic, structural)

- **`sor_juana/local_training.py`**: Local MLX training for Apple Silicon
  - `local_training_pipeline_workflow`: Complete local training pipeline
  - `evaluate_model_workflow`: Batch evaluation on held-out data
  - `generate_sample_workflow`: Generate and evaluate samples
  - MLX-optimized workflows for Apple Silicon Macs

#### Enhanced CLI Commands
- **`sor-juana train`** command group:
  - `train openai`: Start OpenAI fine-tuning with DBOS workflows
  - `train monitor <job_id>`: Monitor fine-tuning jobs with durable execution
  - `train local`: Local MLX-based training and evaluation

- **`sor-juana test`** command group:
  - `test model [model_id]`: Test fine-tuned models with prompts
  - `test model --prompt <prompt>`: Test with custom prompt
  - `test model --compare`: Compare fine-tuned vs base model
  - `test sample`: Generate and evaluate samples with local models

#### Documentation
- **`CHANGELOG.md`**: Version history and release notes with migration guide
- **`scripts/README.md`**: Complete CLI reference and usage guide

### Changed

#### Architecture Refactoring
- **Moved all training scripts to DBOS workflows**:
  - `scripts/train_openai.py` → `sor_juana/training.py` (DBOS workflows)
  - `scripts/test_model.py` → `sor_juana/evaluation.py` (DBOS workflows)
  - `scripts/local-training.py` → `sor_juana/local_training.py` (DBOS workflows)
  - `scripts/monitor_job.py` → Integrated into `sor_juana/training.py`

- **Unified CLI interface**: All training and evaluation now accessible via `sor-juana` CLI
- **Durable execution**: All workflows automatically recover from failures
- **Managed concurrency**: DBOS queues for parallel operations

#### CLI Improvements
- Better error messages and user feedback
- Rich terminal output with progress indicators
- Consistent command structure across all operations
- Automatic DBOS lifecycle management (launch/shutdown)

### Removed

- **`scripts/train_openai.py`**: Replaced by `sor-juana train openai`
- **`scripts/test_model.py`**: Replaced by `sor-juana test model`
- **`scripts/local-training.py`**: Replaced by `sor-juana train local`
- **`scripts/monitor_job.py`**: Functionality integrated into training workflows

### Migration Guide

If you were using the old Python scripts, here's how to migrate to the new CLI commands:

#### OpenAI Fine-Tuning

**Before (v0.1.0):**
```bash
python scripts/train_openai.py
python scripts/monitor_job.py <job_id>
```

**After (v0.2.0):**
```bash
# Single command with automatic monitoring
sor-juana train openai

# Or step by step
sor-juana train openai --no-monitor
sor-juana train monitor <job_id>
```

#### Model Testing

**Before (v0.1.0):**
```bash
python scripts/test_model.py <model_id>
```

**After (v0.2.0):**
```bash
# Test with predefined prompts (uses saved model ID)
sor-juana test model

# Test with specific model ID
sor-juana test model <model_id>

# Test with custom prompt
sor-juana test model --prompt "Your prompt here"

# Compare with base model
sor-juana test model --prompt "Your prompt" --compare
```

#### Local Training (MLX/Apple Silicon)

**Before (v0.1.0):**
```bash
python scripts/local-training.py --csv corpus.csv --eval-only
```

**After (v0.2.0):**
```bash
# Run local training pipeline
sor-juana train local --csv corpus.csv

# Evaluation only (no fine-tuning)
sor-juana train local --eval-only

# Generate and evaluate sample
sor-juana test sample --prompt "Your prompt" --model meta-llama/Llama-2-7b
```

#### Key Changes

- All training and evaluation operations now use DBOS workflows for durable execution
- Workflows automatically resume from the last completed step if interrupted
- Unified CLI interface: `sor-juana train` and `sor-juana test` command groups
- Model IDs are automatically saved to `data/openai_training/fine_tuned_model.json`
- No manual state management needed - DBOS handles persistence automatically

### Fixed

- Workflow recovery: Training jobs now automatically resume if interrupted
- State management: No manual state tracking needed
- Error handling: Better error messages and recovery strategies

### Security

- No security changes in this release

## [0.1.0] - 2024-XX-XX

### Added

#### Core Functionality
- Multi-source text downloading (Gutenberg, Wikisource, BVMC)
- Semantic HTML chunking with metadata preservation
- Intelligent text cleaning and normalization
- MinHash LSH deduplication
- DuckDB storage for efficient corpus management
- Multiple export formats (JSON, JSONL, CSV)
- Train/eval splitting with reproducible seeds

#### DBOS Integration
- `sor_juana/workflows.py`: Corpus building workflows
  - `build_corpus_workflow`: Orchestrated multi-source download
  - `export_csv_workflow`: CSV export with DBOS
  - Concurrent downloads using DBOS queues

#### CLI Interface
- `sor-juana build`: Download and process corpus
- `sor-juana export`: Export to JSON/JSONL
- `sor-juana export-csv`: Export to CSV
- `sor-juana stats`: View corpus statistics
- `sor-juana list`: List texts by source
- `sor-juana split`: Split into train/eval sets
- `sor-juana clear`: Clear corpus database

#### Database
- DuckDB integration for corpus storage
- Efficient querying and filtering
- Source-based organization
- Metadata-rich storage

#### Downloaders
- `GutenbergDownloader`: Project Gutenberg integration
- `WikisourceDownloader`: MediaWiki API client
- `BVMCDownloader`: Biblioteca Virtual Miguel de Cervantes

#### Processing
- `TextDeduplicator`: MinHash LSH implementation
- Semantic HTML parsing
- Text cleaning and normalization
- Chunk extraction with metadata

### Dependencies
- `dbos`: Durable workflow orchestration
- `duckdb`: Embedded analytical database
- `click`: CLI framework
- `rich`: Terminal formatting
- `requests`: HTTP client
- `beautifulsoup4`: HTML parsing
- `mwclient`: MediaWiki API client
- `datasketch`: MinHash LSH
- `nltk`: Natural language processing

---

## Version History

- **0.2.0**: Major refactoring - DBOS workflows for training & evaluation
- **0.1.0**: Initial release - Core corpus building functionality

[Unreleased]: https://github.com/yourusername/sor-night-ws/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yourusername/sor-night-ws/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/sor-night-ws/releases/tag/v0.1.0

