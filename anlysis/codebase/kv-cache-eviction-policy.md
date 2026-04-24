# KV Cache Eviction Policy: vllm v1 and vllm-ascend

**Submodule versions analysed**

| Submodule | Commit |
|-----------|--------|
| `vllm` | `f95c11a848c72c8e10e8c5bb14d0a744f9ae2a82` |
| `vllm-ascend` | `87200ac0b270bf5be555a4a4d8fb57e8d2cb89f4` |

---

## 1. Background: KV cache memory management in vllm v1

vllm v1 organises KV cache as a fixed-size pool of equally-sized **blocks** that live in HBM (GPU/NPU memory). Every block holds the key and value tensors for a fixed number of tokens (the block size). Block management is handled by two main classes:

| Class | File | Responsibility |
|-------|------|---------------|
| `KVCacheBlock` | `vllm/v1/core/kv_cache_utils.py` | Per-block metadata (id, ref-count, hash, free-list links) |
| `FreeKVCacheBlockQueue` | `vllm/v1/core/kv_cache_utils.py` | Doubly-linked list tracking the ordered set of free/evictable blocks |
| `BlockPool` | `vllm/v1/core/block_pool.py` | Owns all blocks, drives allocation and eviction |
| `SingleTypeKVCacheManager` | `vllm/v1/core/single_type_kv_cache_manager.py` | Per-group manager; bridges the scheduler and the block pool |

---

## 2. Core eviction policy: LRU on the free-block queue

### 2.1 The free-block queue

`FreeKVCacheBlockQueue` is a doubly-linked list whose ordering encodes eviction priority:

> "The least recent used block is at the **front** (LRU). If two blocks have the same last accessed time (allocated by the same sequence), the one with more hash tokens (the tail of a block chain) is at the **front**."
>
> — `vllm/v1/core/kv_cache_utils.py`, `FreeKVCacheBlockQueue` docstring

Allocation always pops from the **front** (the coldest block). Freed blocks are **appended to the tail** (the most recently used end). This gives a classic LRU recency ordering.

### 2.2 Block lifecycle and ref-counting

```
ALLOCATED (ref_cnt ≥ 1)
  │
  │  request finishes / is preempted
  │  blocks freed in reverse order (tail first → higher eviction priority)
  ▼
FREE IN QUEUE (ref_cnt == 0)
  ├── [no hash]    → clean free block, reused immediately
  └── [has hash]   → prefix-cached block (eviction candidate)
          │
          │  another request needs a new block and no clean blocks remain
          ▼
      EVICTED from prefix cache hash table
          → hash metadata cleared, HBM bytes reused by new block
```

Key invariants:
- `ref_cnt == 0` is the sole condition for a block to enter the free queue.
- A block in the free queue whose `block_hash` is non-`None` is a **cached** (prefix-cache) block: it is still logically "hot" data in HBM that can be reused without recomputation **until** it gets popped for a new allocation.
- Once popped, `BlockPool._maybe_evict_cached_block()` removes the block's entry from `cached_block_hash_to_block` and resets its hash. The underlying HBM tensor is not zeroed; it is simply overwritten by the next user.

### 2.3 Allocation path

```python
# BlockPool.get_new_blocks()
ret = self.free_block_queue.popleft_n(num_blocks)  # pops LRU blocks
for block in ret:
    self._maybe_evict_cached_block(block)  # removes prefix-cache entry if present
    block.ref_cnt += 1
```

If the pool has fewer than the requested blocks, allocation raises `ValueError`; the **scheduler** is responsible for freeing blocks first (by preempting a running request).

### 2.4 Prefix-cache hit path (touch)

When a new request shares a prefix already cached in HBM:

```python
# BlockPool.touch()
for block in blocks:
    if block.ref_cnt == 0:          # block is in the free queue
        self.free_block_queue.remove(block)  # O(1) removal
    block.ref_cnt += 1              # request now owns a reference
```

Touching a block promotes it from "evictable" to "active": it leaves the free queue and cannot be evicted as long as `ref_cnt > 0`.

---

## 3. What KV cache is wiped from HBM?

Two distinct situations cause KV cache data to be lost from HBM:

### 3.1 Prefix-cache eviction (cold cache blocks)

A **cached** block (one with a block hash, `ref_cnt == 0`, sitting in the free queue) is silently evicted when it reaches the front of the LRU queue and gets reallocated to a new request. The hash metadata is cleared; the block's KV tensor bytes in HBM are overwritten by the new request's forward pass. No explicit zeroing occurs.

**Which blocks?** Any prefix-cached block that has not been accessed recently enough — specifically, the least recently freed blocks, and among those at the same recency level, the ones with the longest token chains (tails of request sequences).

### 3.2 Running-request preemption (scheduler-driven eviction)

When `allocate_slots` in `SingleTypeKVCacheManager` cannot obtain enough free blocks, the scheduler is returned `None` and must preempt a running request:

```python
# vllm/v1/core/sched/scheduler.py  _preempt_request()
self.kv_cache_manager.free(request)    # returns all request blocks to the free queue
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0        # must recompute from the start
self.waiting.prepend_request(request)  # moved to front of the waiting queue
```

`kv_cache_manager.free()` calls `SingleTypeKVCacheManager.free()`, which iterates the request's blocks **in reverse order** (tail first) before calling `BlockPool.free_blocks()`. Reversing ensures that tail blocks (with longer context hashes, which are more expensive to recompute) land at the front of the free queue and are evicted first, while prefix blocks (shorter context) survive longer — a subtle optimization.

**Preemption target selection**:
- FCFS (default): `self.running.pop()` — the request that was **most recently added** to the running queue is preempted first.
- Priority scheduling: `max(self.running, key=lambda r: (r.priority, r.arrival_time))` — the request with the numerically **highest priority value** (lowest business priority) and earliest arrival time is preempted first.

After preemption, the freed blocks may still retain their cached status in `cached_block_hash_to_block` until they are reallocated (at which point eviction happens as in §3.1). This means a newly preempted request's prefix blocks might survive in the LRU cache and serve a cache hit if the request is rescheduled quickly.

---

## 4. vllm-ascend modifications

### 4.1 No change to the core eviction algorithm

vllm-ascend does **not** override or replace `BlockPool`, `FreeKVCacheBlockQueue`, or `SingleTypeKVCacheManager`. The fundamental **LRU eviction policy is inherited unchanged** from vllm.

### 4.2 RecomputeScheduler: modified preemption for PD disaggregation

vllm-ascend introduces `RecomputeScheduler` (`vllm_ascend/core/recompute_scheduler.py`) as the default scheduler (replacing the base `Scheduler`). Its preemption path diverges from the base in one scenario — **when the node is a KV consumer** in a disaggregated prefill/decode (PD) setup:

```python
# RecomputeScheduler._schedule_running() — preemption inner loop
if transfer_config is not None and not transfer_config.is_kv_producer:
    # KV consumer path: instead of moving request to waiting queue,
    # signal the PD proxy to re-prefill on the prefill node
    recomputed_req = self.running.pop()
    self.kv_cache_manager.free(recomputed_req)
    recomputed_reqs.append(
        RecomputeReqInfo(recomputed_req.request_id, ...)
    )
else:
    # Standard path: preempt and move to waiting (same as base scheduler)
    preempted_req = self.running.pop()
    self._preempt_request(preempted_req, scheduled_timestamp)
```

**Effect**: On a decode node in PD disaggregated inference, a preempted request is not re-queued locally; instead, it is entirely dropped from local state and sent back to the prefill node for recomputation. Its HBM blocks are freed immediately (same LRU mechanics), but the request will not be rescheduled locally — it comes back as a fresh request from the PD proxy.

This `RecomputeScheduler` is also used by:
- `scheduler_profiling_chunk.py` — profiling-chunk chunked-prefill scheduler (inherits the same preemption logic)
- `scheduler_dynamic_batch.py` — dynamic batching scheduler (inherits the same preemption logic)

In all three Ascend-specific schedulers the **preemption target selection and LRU eviction mechanics are identical** to the base scheduler.

### 4.3 CPU-NPU KV offloading

vllm-ascend adds `CpuNpuOffloadingHandler` (`vllm_ascend/kv_offload/cpu_npu.py`) which swaps KV blocks between NPU HBM and CPU DRAM using:

- `torch.npu.Stream` — dedicated D2H and H2D streams
- `torch.ops._C_ascend.swap_blocks` — a custom CANN-backed op replacing the CUDA `swap_blocks` kernel

This is wired into vllm's `KVOffload` infrastructure (already present in vllm) and does **not** change when eviction happens — it changes what happens after eviction. Instead of losing the KV data when a prefix-cache block is evicted, the data can first be offloaded to CPU DRAM and reloaded later, avoiding a full recompute. The scheduling of which blocks to offload vs. evict is controlled by vllm's existing offload manager, not by Ascend-specific code.

### 4.4 Summary of ascend differences

| Aspect | vllm base | vllm-ascend |
|--------|-----------|-------------|
| Eviction order | LRU (FreeKVCacheBlockQueue) | **Same** (inherited) |
| Which blocks evicted | Least recently freed, tail-first within a request | **Same** |
| Preemption strategy | Recompute (request → waiting queue) | Recompute (same), **plus** PD consumer mode: request discarded, re-prefill triggered on prefill node |
| HBM→CPU offload | Optional; uses CUDA swap_blocks | Optional; uses NPU-specific `torch.ops._C_ascend.swap_blocks` + `torch.npu.Stream` |
| Explicit zeroing of evicted blocks | No | No |

---

## 5. Key source files

| File | Relevance |
|------|-----------|
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`, `FreeKVCacheBlockQueue` (LRU queue implementation) |
| `vllm/v1/core/block_pool.py` | `BlockPool`: allocation, eviction, prefix-cache, touch |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `FullAttentionManager.free()` — reverse-order block freeing; evictable-block counting |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler._preempt_request()`, preemption target selection |
| `vllm_ascend/core/recompute_scheduler.py` | `RecomputeScheduler` — PD-consumer preemption variant |
| `vllm_ascend/core/scheduler_dynamic_batch.py` | Dynamic-batch scheduler (inherits same preemption) |
| `vllm_ascend/core/scheduler_profiling_chunk.py` | Profiling-chunk scheduler (inherits same preemption) |
| `vllm_ascend/kv_offload/cpu_npu.py` | `CpuNpuOffloadingHandler` — NPU HBM ↔ CPU DRAM swap |
