# Submodule Analysis: `vllm`

**Upstream repo:** https://github.com/vllm-project/vllm  
**Purpose:** High-throughput, memory-efficient LLM inference and serving engine. Originally from UC Berkeley's Sky Computing Lab, now a large community-driven open-source project.

---

## What it does

vLLM implements the full stack for serving large language models at scale:

- Efficient memory management via **PagedAttention** — treating the KV-cache like a virtual memory paging system to avoid memory fragmentation.
- **Continuous batching** of incoming requests so GPU utilization stays high even with variable request lengths.
- **Chunked prefill** to reduce time-to-first-token latency.
- **Prefix caching** for repeated prompt prefixes.
- Multiple **quantization** formats (FP8, MXFP8/4, NVFP4, INT8/4, GPTQ, AWQ, GGUF, compressed-tensors, TorchAO, etc.).
- **Speculative decoding** (n-gram, EAGLE, DFlash, suffix-based).
- **Structured output** generation via xgrammar / guidance.
- **Tool calling** and reasoning parsers.
- Distributed inference with tensor, pipeline, data, expert, and context parallelism.
- **Disaggregated prefill / decode / encode** for flexible cluster topologies.
- OpenAI-compatible REST API, Anthropic Messages API, and gRPC API.
- Multi-LoRA support for dense and MoE layers.

---

## Supported hardware

NVIDIA GPUs (CUDA), AMD GPUs (ROCm/HIP), x86/ARM/PowerPC CPUs, Google TPUs, Intel Gaudi, Huawei Ascend, Apple Silicon, and more (via hardware plugin interface).

---

## Top-level repository layout

| Path | Role |
|------|------|
| `vllm/` | Main Python package (all runtime code lives here) |
| `csrc/` | C++ / CUDA / HIP kernel source compiled into a native extension |
| `cmake/` & `CMakeLists.txt` | Build system for the C++ extension |
| `requirements/` | Split requirements files (build, CPU, CUDA, ROCm, TPU, etc.) |
| `tests/` | Pytest test suite |
| `benchmarks/` | Benchmark scripts for throughput, latency, etc. |
| `examples/` | Usage examples (offline inference, serving, LoRA, multimodal, etc.) |
| `docs/` | MkDocs documentation source |
| `docker/` | Dockerfiles for different hardware targets |
| `scripts/` | Development and release helper scripts |
| `tools/` | CI / tooling utilities |
| `pyproject.toml` / `setup.py` | Python package build (setuptools + setuptools-scm) |

---

## Python package structure (`vllm/`)

### Core engine

| Module | Role |
|--------|------|
| `engine/llm_engine.py` | Synchronous LLM engine — main inference orchestrator |
| `engine/async_llm_engine.py` | Async wrapper for the engine (used by the API server) |
| `sequence.py` | Data model for token sequences and logical/physical blocks |
| `sampling_params.py` | Configuration for sampling (temperature, top-p, beam search, etc.) |
| `outputs.py` | Request/generation output data classes |

### V1 subsystem (`vllm/v1/`)

A redesigned, next-generation engine implementation in active development. Contains its own:

- `engine/` — engine loop and core I/O
- `core/` — scheduler and KV-cache manager for v1
- `worker/` — inference worker process for v1
- `executor/` — multi-process / Ray executor for v1
- `attention/` — attention backends for v1
- `spec_decode/` — speculative decoding for v1
- `structured_output/` — structured output manager
- `kv_offload/` & `simple_kv_offload/` — KV cache offloading to CPU/disk
- `pool/` — memory pool management

### Entrypoints (`vllm/entrypoints/`)

| Sub-path | Role |
|----------|------|
| `openai/` | Full OpenAI-compatible REST API (chat completions, completions, embeddings, etc.) |
| `anthropic/` | Anthropic Messages API compatibility layer |
| `grpc_server.py` | gRPC server |
| `llm.py` | The `LLM` class — the main offline/programmatic entry point |
| `cli/` | Command-line interface (`vllm serve`, `vllm bench`, etc.) |
| `mcp/` | MCP protocol support |
| `pooling/` | Embedding / pooling-mode endpoint |
| `sagemaker/` | AWS SageMaker integration |

### Model executor (`vllm/model_executor/`)

| Sub-path | Role |
|----------|------|
| `models/` | Individual model implementations (~200+ models: Llama, Qwen, DeepSeek, Mixtral, Gemma, BERT, Mamba, etc.) |
| `layers/` | Reusable neural-network layers (attention, linear, fused MoE, quantization, rotary embedding, MLA, layer norm, etc.) |
| `model_loader/` | Loads weights from HuggingFace Hub, Safetensors, GGUF, etc. |
| `kernels/` | Python wrappers around compiled kernels |

### Distributed (`vllm/distributed/`)

| Sub-path | Role |
|----------|------|
| `parallel_state.py` | Process group management (tensor / pipeline / data / expert parallelism) |
| `device_communicators/` | Backend communicators (NCCL, cuCCL, pynccl, shm, etc.) |
| `kv_transfer/` | Cross-node KV-cache transfer for disaggregated serving |
| `eplb/` | Expert-parallel load balancing |
| `elastic_ep/` | Elastic expert parallelism |

### Platform abstraction (`vllm/platforms/`)

Per-hardware platform classes (`cuda.py`, `rocm.py`, `cpu.py`, `tpu.py`, `xpu.py`) implementing a common interface. Hardware plugins (like `vllm-ascend`) register themselves here.

### Compilation (`vllm/compilation/`)

Integration with `torch.compile` for graph-level transformations: CUDA graph capture, piecewise compilation, custom compilation passes, and codegen utilities.

### Other notable modules

| Module | Role |
|--------|------|
| `config/` | Dataclass-based configuration objects for models, cache, scheduling, quantization, etc. |
| `lora/` | Multi-LoRA management and weight application |
| `multimodal/` | Input pipelines for images, audio, video; multimodal processors per model |
| `tokenizers/` | Tokenizer management and caching |
| `spec_decode/` | Speculative decoding engine (legacy v0 path) |
| `reasoning/` | Reasoning-model output parsers |
| `tool_parsers/` | Tool-call parsing for function-calling models |
| `inputs/` | Unified input preprocessing and data pipeline |
| `scheduling/` (v1 `core/`) | Continuous batching scheduler and block allocator |
| `plugins/` | Plugin system for LoRA resolvers and hardware backends |
| `tracing/` | OpenTelemetry tracing support |
| `profiler/` | Profiling utilities (torch profiler integration) |
| `quantization/` (in layers) | Python-side quantization config, weight packing, and linear layers |
| `ir/` | Intermediate representation for model graphs |

---

## C++ / CUDA kernel layer (`csrc/`)

| Sub-path | Role |
|----------|------|
| `attention/` | PagedAttention v1/v2 CUDA kernels, MLA attention, KV-cache merge |
| `moe/` | MoE routing CUDA kernels: top-k softmax, permute/unpermute, GEMM fused kernels, Marlin W×NA16 |
| `quantization/` | Quantization kernels: GPTQ, AWQ, Marlin, Machete, GGUF, W8A8, hadamard |
| `mamba/` | SSM / Mamba selective-scan kernels |
| `cutlass_extensions/` | CUTLASS GEMM extensions for various dtypes |
| `quickreduce/` & `custom_all_reduce.cu` | High-bandwidth intra-node all-reduce |
| `cache_kernels.cu` | KV-cache copy, swap, reshape kernels |
| `layernorm_kernels.cu` | Fused RMSNorm / LayerNorm kernels |
| `activation_kernels.cu` | SiLU, GELU, and other activation fused kernels |
| `pos_encoding_kernels.cu` | RoPE (rotary positional encoding) kernels |
| `sampler.cu` | GPU sampling kernels |
| `cpu/` | CPU-specific implementations (for CPU backend) |
| `rocm/` | ROCm/HIP-specific wrappers |
| `torch_bindings.cpp` | PyTorch custom op registration entry point |

---

## Key technologies and dependencies

| Technology | Use |
|------------|-----|
| Python ≥ 3.10 | Runtime language |
| PyTorch 2.11 | Tensor compute and model execution |
| CUDA / CUTLASS | GPU kernel development |
| Triton | GPU kernel authoring in Python |
| FlashAttention / FlashInfer / TRTLLM-GEN / FlashMLA | Optimized attention implementations |
| HuggingFace Transformers & Hub | Model/tokenizer loading |
| Ray | Multi-node distributed executor |
| FastAPI | REST API server |
| xgrammar / guidance | Structured output generation |
| CMake + ninja | C++ extension build |
| setuptools-scm | Version management from Git tags |
| Apache 2.0 License | |
