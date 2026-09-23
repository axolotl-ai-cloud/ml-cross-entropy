# Test notes

## Chunked classifier accumulation

Two-rank tests cover vocabulary parallelism (including fp32 embedding-gradient
reduction before casting), DDP, and FSDP2 with resharding, mixed precision,
microbatch accumulation, and optimizer updates:

```bash
pytest tests/test_chunked_distributed.py
```

These tests use small models and tensors; they do not establish full-model
training throughput or DeepSpeed ZeRO-3 compatibility.

Boundary tests cover partial final chunks, token and hidden-dimension tile edges,
non-contiguous classifiers, and nonzero storage offsets in FP16/BF16. To detect
out-of-bounds or misaligned CUDA accesses, run the following from the repository
root with a CUDA GPU and Compute Sanitizer available:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. CCE_AUTOTUNE=0 \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
PYTORCH_ALLOC_CONF=expandable_segments:False \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
compute-sanitizer --tool memcheck --padding 256 --error-exitcode 99 \
  --target-processes all python -m pytest -q \
  tests/test_chunked_accumulation.py tests/test_chunked_memory.py \
  tests/test_chunk_recommendation.py
```

Disabling allocator caching and adding guard padding exposes allocation-boundary
errors. Sanitizer errors or failed tests produce a nonzero exit code. Verify that
the GPU tests run rather than skip.
