# OpenAI Fine-Tuning Scripts

Complete toolkit for fine-tuning an OpenAI model to write in the style of Sor Juana Inés de la Cruz.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install openai rich
   ```

2. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. **Build the corpus** (if not done already):
   ```bash
   sor-juana build
   ```

## Scripts

### 1. `train_openai.py` - Main Training Script

Handles the complete fine-tuning pipeline:
- Transforms corpus data to OpenAI's format
- Uploads training and validation files
- Creates a fine-tuning job
- Monitors training progress

**Usage:**
```bash
python scripts/train_openai.py
```

**What it does:**
1. Loads `data/train.jsonl` and `data/eval.jsonl`
2. Transforms them into chat format with proper prompts
3. Uploads to OpenAI
4. Creates a fine-tuning job on `gpt-4o-mini-2024-07-18`
5. Optionally monitors progress until completion

**Output:**
- Transformed files: `data/openai_training/train_openai.jsonl` and `eval_openai.jsonl`
- Model info: `data/openai_training/fine_tuned_model.json`

**Configuration:**

Edit the `FINE_TUNE_CONFIG` dictionary in the script to adjust:
- `model`: Base model to fine-tune
- `n_epochs`: Number of training epochs (default: 3)
- `batch_size`: Batch size (default: auto)
- `learning_rate_multiplier`: Learning rate (default: auto)

### 2. `monitor_job.py` - Monitor Existing Jobs

Check status and monitor an existing fine-tuning job.

**Usage:**
```bash
python scripts/monitor_job.py <job_id>
```

**Example:**
```bash
python scripts/monitor_job.py ftjob-abc123
```

**Features:**
- Shows detailed job status
- Displays recent training events
- Optionally monitors until completion
- Shows final model ID when done

### 3. `test_model.py` - Test Fine-Tuned Model

Generate text using your fine-tuned model and compare with the base model.

**Usage:**
```bash
# Auto-load model from training output
python scripts/test_model.py

# Or specify model ID directly
python scripts/test_model.py ft:gpt-4o-mini-2024-07-18:org:sor-juana:abc123
```

**Test Modes:**

1. **Predefined Prompts**: Run a suite of test prompts covering different genres
2. **Comparison Mode**: Compare fine-tuned vs. base model outputs
3. **Interactive Mode**: Enter custom prompts and generate text in real-time

**Example prompts:**
- "Escribe un soneto sobre la búsqueda del conocimiento"
- "Escribe una reflexión poética sobre la naturaleza del amor"
- "Escribe versos sobre la relación entre fe y razón"

## Complete Workflow

### Step-by-step guide:

```bash
# 1. Set your API key
export OPENAI_API_KEY='sk-...'

# 2. Install dependencies
pip install openai rich

# 3. Build the corpus (if needed)
sor-juana build

# 4. Start fine-tuning
python scripts/train_openai.py

# 5. (Optional) Monitor a running job
python scripts/monitor_job.py ftjob-abc123

# 6. Test your model
python scripts/test_model.py
```

## Data Format

### Input Format (Corpus JSONL)
```json
{
  "text": "Poetic text by Sor Juana...",
  "metadata": {
    "source": "gutenberg",
    "genre": "poetry",
    "title": "..."
  }
}
```

### OpenAI Training Format (Generated)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Eres Sor Juana Inés de la Cruz, poeta y pensadora del siglo XVII..."
    },
    {
      "role": "user",
      "content": "Escribe un poema en tu característico estilo barroco."
    },
    {
      "role": "assistant",
      "content": "Poetic text from corpus..."
    }
  ]
}
```

## Cost Estimation

Fine-tuning costs depend on:
- Number of tokens in training data
- Number of epochs
- Base model used

**Approximate costs** (as of 2024):
- Training: ~$8-15 for gpt-4o-mini with ~500K tokens
- Inference: Same as base model + small fine-tuning fee

Check current pricing: https://openai.com/pricing

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='your-key-here'
```

### "File not found: train.jsonl"
Make sure to build the corpus first:
```bash
sor-juana build
```

### "Fine-tuning failed"
Check the job status for error details:
```bash
python scripts/monitor_job.py <job_id>
```

Common issues:
- Training file too small (need >10 examples)
- Malformed JSONL
- Insufficient API credits

### "Missing dependencies"
```bash
pip install openai rich
```

## Advanced Usage

### Custom System Prompt

Edit the `system_message` in `train_openai.py` (line 91):

```python
system_message = (
    "Your custom system message here..."
)
```

### Custom User Prompts

Modify the prompt generation logic in `transform_to_openai_format()` (lines 102-110):

```python
# Add genre-specific or custom prompts
if "poetry" in genre.lower():
    user_prompt = "Your custom prompt..."
```

### Hyperparameter Tuning

Adjust in `train_openai.py`:

```python
FINE_TUNE_CONFIG = {
    "model": "gpt-4o-mini-2024-07-18",
    "n_epochs": 5,  # More epochs for better learning
    "batch_size": 4,  # Smaller batch for more updates
    "learning_rate_multiplier": 1.5,  # Faster learning
}
```

## Resources

- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/fine-tuning)
- [Best Practices for Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)

## Support

For issues specific to these scripts, please check:
1. The script's inline comments
2. This README
3. OpenAI's documentation
4. The project's main README

---

**Happy fine-tuning!** 🎨📚
