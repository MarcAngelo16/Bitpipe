# BitPipe Migration Plan

## Phase 1: Directory Structure ✅ COMPLETED

Created the following directory structure:
- `schedules/` - Organized schedule implementations
- `schedules/bitpipe/` - BitPipe family (4-VR)
- `schedules/chimera/` - Chimera family (2-VR) [Future]
- `asymmetric/` - Asymmetric configuration management

All directories have `__init__.py` files with proper documentation.

---

## Phase 2: Migrate BitPipe Code (NEXT)

### Files to Create:

1. **schedules/bitpipe/bitpipe_4vr.py**
   - Copy from: `/workspace/Bitpipe/megatron/core/pipeline_parallel/bitpipe_schedule.py`
   - Main function: `forward_backward_pipelining_with_BitPipe` → rename to `forward_backward_pipelining_with_bitpipe_4vr`
   - Keep all helper functions and logic
   - Update imports

2. **schedules/bitpipe/bitpipe_utils.py**
   - Extract shared utilities if needed
   - For now, keep everything in bitpipe_4vr.py for simplicity

### Files to Update:

1. **schedules.py**
   - Update import path:
     ```python
     # Old:
     from megatron.core.pipeline_parallel.bitpipe_schedule import forward_backward_pipelining_with_BitPipe

     # New:
     from megatron.core.pipeline_parallel.schedules.bitpipe.bitpipe_4vr import forward_backward_pipelining_with_bitpipe_4vr
     ```

2. **schedules/bitpipe/__init__.py**
   - Export the main function:
     ```python
     from .bitpipe_4vr import forward_backward_pipelining_with_bitpipe_4vr
     __all__ = ['forward_backward_pipelining_with_bitpipe_4vr']
     ```

### Backward Compatibility:

Keep `bitpipe_schedule.py` as a compatibility shim:
```python
# Backward compatibility - import from new location
from megatron.core.pipeline_parallel.schedules.bitpipe.bitpipe_4vr import (
    forward_backward_pipelining_with_bitpipe_4vr as forward_backward_pipelining_with_BitPipe
)

# Deprecation warning
import warnings
warnings.warn(
    "Importing from bitpipe_schedule.py is deprecated. "
    "Please use: from megatron.core.pipeline_parallel.schedules.bitpipe import forward_backward_pipelining_with_bitpipe_4vr",
    DeprecationWarning,
    stacklevel=2
)
```

---

## Phase 3: Test Migrated BitPipe

### Test Cases:

1. **Functional Test**: Run existing training script
   - Verify output matches original
   - Check no import errors
   - Verify backward compatibility

2. **Performance Test**: Compare performance metrics
   - Throughput (samples/sec)
   - Memory usage
   - Pipeline bubble time
   - Communication overhead

3. **Profiling Test**: Use bitpipe_profiler
   - Enable: `--enable-bitpipe-profiling`
   - Compare profile outputs

### Commands to Test:

```bash
# Test with symmetric BitPipe
torchrun pretrain_gpt.py \
    --enable-bitpipe-schedule \
    --pipeline-model-parallel-size 4 \
    --num-layers 16 \
    --micro-batch-size 2 \
    --global-batch-size 8

# Test with asymmetric BitPipe (if available)
torchrun pretrain_gpt.py \
    --enable-bitpipe-schedule \
    --enable-bitpipe-asymmetric \
    --bitpipe-asymmetric-config configs/4_devices/asymmetric_4devices_24layers.json \
    --pipeline-model-parallel-size 4 \
    --num-layers 24
```

---

## Phase 4: Migrate Asymmetric Support

### Files to Create:

1. **asymmetric/config_loader.py**
   - Move from: `/workspace/Bitpipe/megatron/bitpipe_asymmetric_utils.py`
   - Function: `generate_asymmetric_config_from_user_input`
   - Add: `AsymmetricConfig4VR` class

2. **asymmetric/layer_assignment.py**
   - Move from: `/workspace/Bitpipe/megatron/bitpipe_asymmetric_utils.py`
   - Function: `get_asymmetric_offset` → rename to `get_asymmetric_offset_4vr`

3. **asymmetric/validation.py**
   - Add validation logic for asymmetric configs

### Update References:

- `transformer.py`: Update imports to use new asymmetric module
- `bitpipe_4vr.py`: Update imports

---

## Phase 5: Implement Chimera (Future)

Will be designed after BitPipe migration is complete and tested.

---

## Phase 6: Documentation and Cleanup

1. Update CLAUDE.md with new architecture
2. Update README.md with new import paths
3. Add migration guide for users
4. Remove old files (after confirming everything works)
