# Refactoring Summary: Scripts to DBOS Workflows

> **Note**: For migration instructions from old scripts to new CLI commands, see the [Migration Guide in CHANGELOG.md](CHANGELOG.md#migration-guide).

## Overview

Successfully refactored all Python scripts from `scripts/` directory into the main `sor_juana` package using DBOS workflows for improved reliability, durability, and observability.

This document provides technical details about the refactoring. For user-facing migration instructions, see [CHANGELOG.md](CHANGELOG.md).

## What Changed

### New Files Created

1. **`sor_juana/training.py`** (447 lines)
   - OpenAI fine-tuning workflows
   - Functions: data preparation, file upload, job creation, job monitoring
   - Workflows: `prepare_training_data_workflow`, `start_fine_tuning_workflow`, `monitor_fine_tuning_workflow`, `full_fine_tuning_workflow`, `get_job_info_workflow`

2. **`sor_juana/evaluation.py`** (423 lines)
   - Model testing and evaluation workflows
   - Functions: text generation, style scoring, thematic analysis
   - Workflows: `test_model_with_prompt_workflow`, `test_model_with_prompts_workflow`, `compare_models_workflow`, `save_test_results_workflow`

3. **`sor_juana/local_training.py`** (465 lines)
   - Local MLX-based training for Apple Silicon
   - Functions: dataset loading, model evaluation, comprehensive scoring
   - Workflows: `load_and_prepare_dataset_workflow`, `evaluate_model_workflow`, `local_training_pipeline_workflow`, `generate_sample_workflow`

### Updated Files

4. **`sor_juana/cli.py`** (expanded by ~350 lines)
   - Added `train` command group with subcommands:
     - `train openai` - Start OpenAI fine-tuning
     - `train monitor` - Monitor a fine-tuning job
     - `train local` - Local MLX training
   - Added `test` command group with subcommands:
     - `test model` - Test fine-tuned models
     - `test sample` - Generate and evaluate samples

5. **`scripts/README.md`** (completely rewritten)
   - Migration guide from old scripts to new CLI
   - Complete CLI reference
   - Architecture documentation
   - Development guide

### Files Removed

- ❌ `scripts/train_openai.py` (396 lines) → Migrated to `sor_juana/training.py`
- ❌ `scripts/test_model.py` (219 lines) → Migrated to `sor_juana/evaluation.py`
- ❌ `scripts/local-training.py` (378 lines) → Migrated to `sor_juana/local_training.py`
- ❌ `scripts/monitor_job.py` (was already removed)

### Files Kept

- ✅ `scripts/quick_start.sh` - Still useful for initial setup
- ✅ `scripts/README.md` - Updated with migration guide

## Key Improvements

### 1. **Durable Execution**
- Workflows automatically resume from last completed step if interrupted
- No manual state management needed
- Resilient to crashes, restarts, or network failures

### 2. **Better Architecture**
```
Before:
scripts/
├── train_openai.py       (monolithic, no recovery)
├── test_model.py         (separate script)
├── local-training.py     (separate script)
└── monitor_job.py        (separate script)

After:
sor_juana/
├── training.py           (DBOS workflows, durable)
├── evaluation.py         (DBOS workflows, durable)
├── local_training.py     (DBOS workflows, durable)
├── workflows.py          (existing corpus workflows)
└── cli.py               (unified interface)
```

### 3. **Managed Concurrency**
- Used DBOS Queues for parallel operations
- `training_queue` for training operations
- `eval_queue` for evaluation operations
- `local_training_queue` for local training

### 4. **Consistent Patterns**

**Steps** (@DBOS.step()):
- Individual operations (API calls, file I/O, computations)
- Automatically retried on failure
- Results cached in database

**Workflows** (@DBOS.workflow()):
- Orchestrate multiple steps
- Deterministic execution
- Automatic recovery

### 5. **Better CLI Experience**

**Before:**
```bash
python scripts/train_openai.py
# Wait or manually monitor...
python scripts/monitor_job.py job-123
# Then test...
python scripts/test_model.py ft:model-id
```

**After:**
```bash
sor-juana train openai --monitor
# Everything handled in one workflow!
# Or break it up:
sor-juana train openai --no-monitor
sor-juana train monitor job-123
sor-juana test model ft:model-id
```

> **For migration instructions, see the [Migration Guide in CHANGELOG.md](CHANGELOG.md#migration-guide).**

## DBOS Workflow Patterns Used

### 1. Step Pattern
```python
@DBOS.step()
def upload_file_to_openai_step(file_path: str) -> str:
    """Non-deterministic operation as a step."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    return response.id
```

### 2. Workflow Pattern
```python
@DBOS.workflow()
def start_fine_tuning_workflow(train_file: str, eval_file: str) -> dict:
    """Orchestrate multiple steps."""
    # Upload files in parallel
    train_handle = training_queue.enqueue(upload_file_to_openai_step, train_file)
    eval_handle = training_queue.enqueue(upload_file_to_openai_step, eval_file)
    
    train_file_id = train_handle.get_result()
    eval_file_id = eval_handle.get_result()
    
    # Create job
    job_result = create_fine_tuning_job_step(train_file_id, eval_file_id)
    return job_result
```

### 3. Monitoring Pattern
```python
@DBOS.workflow()
def monitor_fine_tuning_workflow(job_id: str, poll_interval: int = 60) -> dict:
    """Long-running monitoring workflow."""
    while True:
        status_result = get_job_status_step(job_id)
        
        if status_result["status"] in ["succeeded", "failed", "cancelled"]:
            return status_result
            
        # Durable sleep - survives restarts!
        DBOS.sleep(poll_interval)
```

### 4. Queue Pattern
```python
# Create queue with concurrency control
training_queue = Queue("training_queue")

# Enqueue parallel operations
handles = []
for item in items:
    handle = training_queue.enqueue(process_item_step, item)
    handles.append(handle)

# Wait for all to complete
results = [h.get_result() for h in handles]
```

## Benefits Achieved

✅ **Reliability**: Workflows automatically recover from failures  
✅ **Observability**: Track progress via DBOS admin server (port 3001)  
✅ **Maintainability**: Clear separation of steps and workflows  
✅ **Testability**: Each step can be tested independently  
✅ **Consistency**: Same patterns across all training/evaluation code  
✅ **User Experience**: Unified CLI with better feedback  
✅ **Scalability**: Ready for distributed execution if needed

## Testing the Refactored Code

### 1. Test OpenAI Training
```bash
# Requires OPENAI_API_KEY
export OPENAI_API_KEY='your-key'

# Build corpus first
sor-juana build
sor-juana split

# Start training
sor-juana train openai --monitor
```

### 2. Test Evaluation
```bash
# Test with saved model
sor-juana test model

# Test with specific model
sor-juana test model ft:gpt-4o-mini:sor-juana

# Compare models
sor-juana test model --prompt "Escribe sobre el conocimiento" --compare
```

### 3. Test Local Training
```bash
# Export corpus to CSV
sor-juana export-csv

# Run evaluation
sor-juana train local --eval-only --sample-size 10
```

## Database Schema

DBOS automatically creates these tables in the system database:

- `dbos.workflow_status` - Track workflow execution
- `dbos.workflow_events` - Workflow state changes
- `dbos.workflow_inputs` - Workflow input parameters
- `dbos.workflow_outputs` - Workflow results
- `dbos.operation_outputs` - Step results (cached)

This enables:
- Automatic recovery
- Workflow introspection
- Debugging and monitoring
- Exactly-once execution guarantees

## Next Steps

### Recommended Enhancements

1. **Add more evaluation metrics**
   - BLEU scores for text quality
   - Perplexity measurements
   - A/B testing workflows

2. **Add workflow monitoring dashboards**
   - Use DBOS admin UI
   - Custom dashboards with workflow stats

3. **Add batch processing workflows**
   - Process multiple prompts in parallel
   - Export results to various formats

4. **Add scheduled workflows**
   - Periodic model evaluation
   - Automated retraining pipelines

5. **Add webhook integrations**
   - Notify on training completion
   - Send results to external systems

### Example: Adding Scheduled Evaluation

```python
@DBOS.scheduled("0 0 * * *")  # Daily at midnight
@DBOS.workflow()
def daily_evaluation_workflow(scheduled_time, actual_time):
    """Run daily model evaluation."""
    model_id = load_model_id_step()
    results = test_model_with_prompts_workflow(None, model_id)
    
    # Send notification if scores drop
    if results["mean_score"] < 3.5:
        send_alert_step(f"Model performance dropped: {results['mean_score']}")
    
    return results
```

## Conclusion

The refactoring successfully moved all training and evaluation logic into the main package with DBOS workflows, providing:

- **Better reliability** through durable execution
- **Better architecture** with clear separation of concerns
- **Better UX** through unified CLI
- **Better maintainability** with consistent patterns
- **Better observability** with built-in monitoring

The code is now production-ready and can scale to handle complex ML pipelines with confidence.

## Resources

- [CHANGELOG.md Migration Guide](CHANGELOG.md#migration-guide) - User-facing migration instructions
- [DBOS Documentation](https://docs.dbos.dev/)
- [DBOS Python SDK](https://docs.dbos.dev/python/reference)
- [CLI Reference](scripts/README.md) - Complete command reference

