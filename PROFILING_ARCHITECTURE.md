# BitPipe vs Standard 1F1B Profiling Architecture

## Key Differences in Profiling Data

### BitPipe Schedule
- **Pipeline ID**: 0 or 1 (bidirectional pipelines)
- **Model Chunk ID**: 0, 1, 2, 3 (4 model chunks total)
- **Pipeline Structure**: Each rank can execute microbatches from both pipelines
- **Execution Pattern**: Interleaved execution between two virtual pipelines

Example BitPipe profiling call:
```python
pipeline_id = 0 if model_chunk_id < 2 else 1  # Pipeline based on chunk
profiler.start_microbatch(microbatch_id, pipeline_id, model_chunk_id, 'forward')
```

### Standard 1F1B Schedule
- **Pipeline ID**: Always 0 (unidirectional pipeline)  
- **Model Chunk ID**: Equals rank (each rank handles one model chunk)
- **Pipeline Structure**: Sequential execution, each rank handles one part of model
- **Execution Pattern**: Classic 1F1B pattern (warmup → steady-state → cooldown)

Example 1F1B profiling call:
```python
profiler.start_microbatch(microbatch_id, 0, rank, 'forward')  # pipeline_id=0, chunk_id=rank
```

## Data Interpretation

### BitPipe Data
- `model_chunk_id` ranges from 0-3 across all ranks
- Multiple `pipeline_id` values (0, 1) per rank
- Complex interleaving patterns visible in timeline

### Standard 1F1B Data  
- `model_chunk_id` equals rank (0, 1, 2, 3 for 4-rank setup)
- Single `pipeline_id` (always 0) for all events
- Sequential pattern: warmup → 1F1B alternating → cooldown

## Analysis Implications

This architecture difference allows us to:
1. **Correctly compare** pipeline efficiency between schedules
2. **Visualize** the different execution patterns accurately  
3. **Identify** where each rank spends time in each schedule
4. **Understand** communication patterns specific to each approach

The profiling data now accurately reflects the underlying pipeline architecture!