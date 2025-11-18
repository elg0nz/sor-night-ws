# Sor Juana Scripts

This directory contains utility scripts for the Sor Juana project.

## Available Scripts

### `quick_start.sh`

Quick start script for setting up and running the project.

```bash
./scripts/quick_start.sh
```

This script:
1. Installs dependencies
2. Downloads and builds the corpus
3. Splits data into train/eval sets
4. Provides next steps for training

## CLI Reference

All training and evaluation functionality is available through the `sor-juana` CLI command, which uses DBOS workflows for reliable, durable execution.

### Corpus Management

```bash
# Build corpus from all sources
sor-juana build

# Export to different formats
sor-juana export --format jsonl
sor-juana export-csv

# View statistics
sor-juana stats

# List texts
sor-juana list --source all

# Split into train/eval
sor-juana split --eval-ratio 0.15

# Clear corpus
sor-juana clear
```

### Training

```bash
# OpenAI fine-tuning
sor-juana train openai [OPTIONS]
  --train-file PATH       Training JSONL file (default: data/train.jsonl)
  --eval-file PATH        Evaluation JSONL file (default: data/eval.jsonl)
  --model TEXT           Base model (default: gpt-4o-mini-2024-07-18)
  --suffix TEXT          Model name suffix (default: sor-juana)
  --monitor/--no-monitor  Monitor progress (default: monitor)

# Monitor job
sor-juana train monitor JOB_ID [OPTIONS]
  --poll-interval INTEGER  Seconds between checks (default: 60)

# Local training (Apple Silicon/MLX)
sor-juana train local [OPTIONS]
  --csv PATH              CSV file with prompt/completion pairs
  --model TEXT           Base model (default: meta-llama/Llama-2-7b)
  --output PATH          Output directory (default: ./sor_juana_model)
  --eval-only            Skip training, only evaluate
  --sample-size INTEGER  Number of examples to evaluate (default: 20)
```

### Testing & Evaluation

```bash
# Test model with predefined prompts
sor-juana test model [MODEL_ID]

# Test with custom prompt
sor-juana test model [MODEL_ID] --prompt "Your prompt here"

# Compare with base model
sor-juana test model [MODEL_ID] --prompt "Your prompt" --compare

# Generate sample with local model
sor-juana test sample --prompt "Your prompt" [OPTIONS]
  --model TEXT  Model name (default: meta-llama/Llama-2-7b)
```

## Architecture

The application uses DBOS workflows for all training and evaluation operations:

```
sor_juana/
├── training.py         # OpenAI fine-tuning workflows
├── evaluation.py       # Model testing & evaluation workflows  
├── local_training.py   # Local MLX training workflows
├── workflows.py        # Corpus building workflows
└── cli.py             # Unified CLI interface
```

Each module contains:
- **@DBOS.step()**: Individual operations (API calls, file operations, etc.)
- **@DBOS.workflow()**: Orchestrated multi-step processes
- **Queue**: Managed concurrency for parallel operations

### DBOS Workflow Benefits

- **Durable Execution**: Workflows automatically resume from the last completed step if interrupted
- **Reliability**: Built-in retry logic and error handling
- **Observability**: Track workflow progress and status via admin server (port 3001)
- **Concurrency**: Managed parallel execution with queues
- **State Management**: Automatic state persistence in the database

## Environment Variables

Make sure to set required environment variables:

```bash
# For OpenAI fine-tuning and testing
export OPENAI_API_KEY='your-api-key-here'

# Optional: DBOS system database URL
export DBOS_SYSTEM_DATABASE_URL='postgresql://user:pass@host:port/dbname'
```

## Example Workflow

Complete workflow from corpus to fine-tuned model:

```bash
# 1. Build the corpus
sor-juana build

# 2. Split into train/eval sets
sor-juana split --eval-ratio 0.15

# 3. Start OpenAI fine-tuning
sor-juana train openai --monitor

# 4. Test the fine-tuned model
sor-juana test model

# 5. Compare with base model
sor-juana test model --prompt "Escribe un soneto sobre el conocimiento" --compare
```

## Development

The DBOS architecture makes it easy to extend functionality:

1. **Add new workflows**: Create new `@DBOS.workflow()` functions
2. **Add new steps**: Create new `@DBOS.step()` functions
3. **Extend CLI**: Add new commands in `cli.py`
4. **Monitor workflows**: Use DBOS admin server (runs on port 3001 by default)

### Example: Adding a New Workflow

```python
# In sor_juana/training.py

@DBOS.step()
def my_new_step(param: str) -> dict:
    """Step description."""
    # Your code here
    return {"result": "success"}

@DBOS.workflow()
def my_new_workflow(input_data: str) -> dict:
    """Workflow description."""
    result = my_new_step(input_data)
    return result
```

Then add a CLI command in `cli.py`:

```python
@train.command(name="my-command")
@click.option("--input", required=True)
def my_command(input: str):
    """Command description."""
    from .training import my_new_workflow
    
    launch_dbos()
    result = my_new_workflow(input)
    console.print(f"Result: {result}")
    shutdown_dbos()
```

## Resources

- [DBOS Documentation](https://docs.dbos.dev/)
- [DBOS Python Workflows](https://docs.dbos.dev/python/tutorials/workflow-tutorial)
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [MLX Documentation](https://ml-explore.github.io/mlx/)

## Support

For issues or questions:
1. Check the main project README
2. Review DBOS documentation
3. Check workflow status: `http://localhost:3001` (DBOS admin server)
