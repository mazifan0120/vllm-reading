# Submodule Analysis: `vllm-ascend`

**Upstream repo:** https://github.com/vllm-project/vllm-ascend  
**Purpose:** Hardware plugin that enables vLLM to run seamlessly on Huawei Ascend NPUs. Implements the vLLM hardware-pluggable interface defined in [RFC #11162](https://github.com/vllm-project/vllm/issues/11162).

---

## What it does

`vllm-ascend` is a community-maintained hardware backend plugin. Rather than forking vLLM and maintaining a separate branch, it extends vLLM via its plugin mechanism so that all Ascend-specific code is isolated here, and consumers just install both packages together.

Key capabilities it brings to vLLM on Ascend hardware:

- Ascend NPU platform registration and detection
- NPU-specific attention kernels (standard, MLA, sparse flash attention)
- NPU-specific fused MoE routing, dispatch, and combine kernels
- Quantization support adapted to CANN ops
- LoRA support on NPU
- Speculative decoding on NPU
- Distributed inference (tensor parallel, expert parallel) using torch-npu collective ops
- KV-cache offloading to CPU
- Multi-Level Attention (MLA) for DeepSeek-style models
- Expert-parallel load balancing (EPLB)
- Disaggregated prefill / decode
- Model loading adapted for NPU memory layout

---

## Supported hardware

| Series | Role |
|--------|------|
| Atlas 800I A2 / A3 Inference series | Primary inference target |
| Atlas A2 / A3 Training series | Supported |
| Atlas 300I Duo | Experimental |

**OS:** Linux only  
**CANN version:** 8.5.1 (Ascend SDK/HDK)  
**PyTorch:** 2.9.0 + torch-npu 2.9.0

---

## Top-level repository layout

| Path | Role |
|------|------|
| `vllm_ascend/` | Main Python package |
| `csrc/` | C++ / CANN operator source compiled as a native extension |
| `cmake/` & `CMakeLists.txt` | Build system for the C++ extension |
| `requirements.txt` / `requirements-dev.txt` | Runtime and dev dependencies |
| `tests/` | Test suite |
| `benchmarks/` | Benchmark scripts |
| `examples/` | Usage examples |
| `docs/` | Documentation source |
| `Dockerfile*` | Multiple Dockerfiles for different Atlas hardware generations |
| `pyproject.toml` / `setup.py` | Package build config |
| `collect_env.py` | Environment diagnostics helper |

---

## Python package structure (`vllm_ascend/`)

### Platform registration

| File | Role |
|------|------|
| `platform.py` | Registers the Ascend NPU as a vLLM platform. This is the entry point vLLM's platform discovery calls. |
| `meta_registration.py` | Registers Ascend-specific custom ops and patches into vLLM's plugin system |
| `ascend_config.py` | Ascend-specific configuration dataclasses (e.g., expert parallelism settings, quantization knobs) |
| `envs.py` | Ascend-specific environment variable handling |
| `ascend_forward_context.py` | Forward pass context management for Ascend |

### Attention (`vllm_ascend/attention/`)

| File | Role |
|------|------|
| `attention_v1.py` | Main attention backend for vLLM v1 path |
| `mla_v1.py` | Multi-Level Attention (MLA) backend for DeepSeek-V2/V3-style models |
| `sfa_v1.py` | Sparse flash attention backend |
| `attention_mask.py` | Ascend-specific attention mask construction |
| `context_parallel/` | Context parallelism support for long-context inference |

### Operators / layers (`vllm_ascend/ops/`)

Ascend-specific implementations of vLLM's neural-network layer interfaces:

| File | Role |
|------|------|
| `activation.py` | Activation functions via CANN ops |
| `layernorm.py` | RMSNorm / LayerNorm using CANN fused kernels |
| `linear.py` / `linear_op.py` | Distributed linear layer adapted for NPU memory layout |
| `rotary_embedding.py` | RoPE implementation on NPU |
| `vocab_parallel_embedding.py` | Vocabulary embedding with tensor parallelism |
| `fused_moe/` | Fused MoE forward pass: routing, dispatch, grouped GEMM, combine |
| `mla.py` | MLA projection ops |
| `qwen2_decoder.py` | Qwen2-specific fused decoder op |
| `triton/` | Triton-Ascend kernel implementations |

### Worker (`vllm_ascend/worker/`)

| File | Role |
|------|------|
| `worker.py` | Main inference worker process for Ascend |
| `model_runner_v1.py` | Model execution loop for v1 engine path |
| `npu_input_batch.py` | Input batching adapted for NPU memory layout |
| `block_table.py` | KV-cache block table management |
| `v2/` | v2 worker variants |

### Distributed (`vllm_ascend/distributed/`)

Distributed communication primitives via torch-npu collective ops and HCCL (Huawei Collective Communication Library). Covers tensor-parallel and expert-parallel collectives.

### Core (`vllm_ascend/core/`)

Scheduler and block allocator overrides for Ascend-specific constraints.

### Quantization (`vllm_ascend/quantization/`)

Quantization schemes adapted to CANN's supported data types and operator library.

### Compilation (`vllm_ascend/compilation/`)

Ascend-specific `torch.compile` passes and graph transformations (mirrors vLLM's compilation module but tuned for NPU graph execution).

### Device / memory (`vllm_ascend/device/`, `vllm_ascend/device_allocator/`)

NPU device management, memory allocator, and CANN memory handling.

### Speculative decoding (`vllm_ascend/spec_decode/`)

Speculative decoding adapted for Ascend hardware.

### KV cache offload (`vllm_ascend/kv_offload/`)

Offloading KV cache blocks to host CPU memory.

### LoRA (`vllm_ascend/lora/`)

LoRA adapter application on NPU (both dense and MoE layers).

### Expert-parallel load balancing (`vllm_ascend/eplb/`)

Dynamic load-balancing of MoE expert assignment across NPU devices.

### Model loader (`vllm_ascend/model_loader/`)

Model weight loading adapted for Ascend (handles CANN format requirements and NPU memory layout).

### Patch system (`vllm_ascend/patch/`)

Runtime monkey-patching of vLLM internals where hardware differences require behavioural changes rather than simple layer replacement.

### Hardware-specific subsets

| Path | Role |
|------|------|
| `_310p/` | Code specific to Atlas 300I Duo (Experimental) |
| `_cann_ops_custom/` | Custom CANN operator wrappers |
| `xlite/` | Lite/edge NPU support |

---

## C++ / CANN operator layer (`csrc/`)

Unlike vLLM's CUDA kernels, these are written against the CANN (Compute Architecture for Neural Networks) API — Huawei's NPU programming framework — and bound into Python via pybind11.

| Sub-path | Role |
|----------|------|
| `aclnn_torch_adapter/` | Adapter layer bridging CANN's ACLNN API and PyTorch custom ops |
| `attention/` (sparse_flash_attention) | Sparse flash attention kernel implementation for NPU |
| `moe_gating_top_k/`, `moe_grouped_matmul/`, `moe_dispatch_normal/`, `moe_combine_normal/` | MoE top-k gating, grouped GEMM, dispatch/combine kernels |
| `dispatch_ffn_combine*/` | Fused FFN dispatch-combine kernels (BF16, W4A8 variants) |
| `matmul_allreduce_add_rmsnorm/` | Fused matmul + all-reduce + RMSNorm kernel |
| `add_rms_norm_bias/` | Fused add + RMSNorm + bias kernel |
| `mla_preprocess/` | MLA query/key preprocessing kernel |
| `grouped_matmul_swiglu_quant_weight_nz_tensor_list/` | Fused grouped GEMM + SwiGLU + quantization for MoE FFN |
| `causal_conv1d/`, `causal_conv1d_v310/` | Causal 1D convolution for Mamba/SSM models |
| `recurrent_gated_delta_rule_v310/` | Delta-rule SSM kernel for 310p variant |
| `apply_top_k_top_p_custom/` | Custom top-k / top-p sampling kernel |
| `reshape_and_cache_bnsd/` | KV-cache reshape into BNSD layout |
| `transpose_kv_cache_by_block/` | Block-level KV-cache transpose |
| `copy_and_expand_eagle_inputs/` | EAGLE speculative decoding input preparation |
| `notify_dispatch/` | Dispatch synchronization primitives |
| `batch_matmul_transpose/` | Batched matmul with transpose |
| `camem_allocator.cpp` | CANN memory allocator wrapper |
| `torch_binding.cpp` / `torch_binding_meta.cpp` | PyTorch op registration |

---

## How it integrates with vLLM

1. **Installation side-by-side** with vLLM (same version).
2. On import, `vllm_ascend.__init__` registers the Ascend platform via vLLM's `VLLM_PLUGINS` entry-point or explicit `register_plugin()` call.
3. vLLM then queries the active platform for its implementations of attention backends, workers, distributed communicators, and compiler passes — and calls the Ascend versions if the runtime platform is Ascend.
4. The monkey-patch system (`patch/`) fills any gaps where vLLM doesn't have a clean plugin hook yet.

---

## Key technologies and dependencies

| Technology | Use |
|------------|-----|
| Python ≥ 3.10, < 3.12 | Runtime language |
| PyTorch 2.9.0 + torch-npu 2.9.0 | NPU tensor compute |
| CANN 8.5.1 | Ascend hardware SDK (operator library, memory management) |
| HCCL | Huawei's collective communication library (like NCCL for Ascend) |
| Triton-Ascend | Triton fork adapted for NPU kernel authoring |
| pybind11 | Python bindings for C++ CANN operators |
| CMake + ninja | C++ extension build |
| xgrammar | Structured output generation |
| Apache 2.0 License | |
