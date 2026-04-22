# `rpc_lookup_port` in KV connector configs: design purpose and usage

## TL;DR

- The actual active key is **`lookup_rpc_port`** (not `rpc_lookup_port`) in `kv_connector_extra_config`.
- It is used by **vLLM-Ascend KV Pool (AscendStoreConnector)** to choose the local ZMQ lookup RPC endpoint between scheduler and worker-side lookup server.
- In code, `lookup_rpc_port` is preferred; legacy `mooncake_rpc_port` is still accepted with a deprecation warning.
- In upstream `vllm` submodule, this key does **not** exist; it is currently a **vllm-ascend-specific** KV Pool setting.

---

## 1) What was found in each submodule

### A. `vllm` submodule

- No `rpc_lookup_port`, `lookup_rpc_port`, or `mooncake_rpc_port` usage was found.
- `vllm` has generic `kv_transfer_config` / `kv_connector_extra_config`, but this specific port key is not part of core upstream connectors.

### B. `vllm-ascend` submodule

- `lookup_rpc_port` is documented and implemented for KV Pool.
- Core behavior is implemented in:
  - `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
  - `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
  - docs: `docs/source/user_guide/feature_guide/kv_pool.md`

---

## 2) Design purpose

`lookup_rpc_port` namespaces the lookup RPC channel used by KV Pool scheduler/worker communication.

Implementation details:

1. `LookupKeyServer` (worker side, rank 0) binds a ZMQ `REP` socket.
2. `LookupKeyClient` (scheduler side) connects via ZMQ `REQ`.
3. Both sides call `get_zmq_rpc_path_lookup(vllm_config)` to derive the same endpoint.
4. Endpoint format is:

`ipc://{VLLM_RPC_BASE_PATH}/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}`

So this field is effectively an endpoint discriminator that prevents collisions when multiple instances are deployed (especially PD-colocated / multi-instance scenarios).

---

## 3) Current runtime behavior

`get_zmq_rpc_path_lookup(...)` logic:

- default `rpc_port = 0`
- if `kv_connector_extra_config["lookup_rpc_port"]` exists, use it
- else if `kv_connector_extra_config["mooncake_rpc_port"]` exists, use it and emit warning:
  - recommend `lookup_rpc_port`
  - `mooncake_rpc_port` planned for removal

This gives backward compatibility while migrating old configs.

---

## 4) Function call chain

This section traces how `lookup_rpc_port` is actually exercised at runtime, on both the **startup** (port resolution) path and the **per-request** (lookup query) path.

### 4.1 Startup path — port resolution

Both the server and the client call the same helper to derive the IPC socket address. The config key is read **once** during connector initialization, long before any request arrives.

```
# ---- WORKER PROCESS (rank 0) ----

vllm serve ...
  └─ EngineCore.__init__()  [vllm/vllm/v1/engine/core.py]
       └─ KVConnectorBase_V1 subclass instantiated with role=WORKER
            └─ AscendStoreConnector.__init__(vllm_config, role=WORKER)
                 │   [vllm-ascend/…/ascend_store_connector.py]
                 ├─ KVPoolWorker.__init__(vllm_config, use_layerwise)
                 │    [vllm-ascend/…/pool_worker.py]
                 └─ LookupKeyServer.__init__(pool_worker, vllm_config, use_layerwise)
                      │  [vllm-ascend/…/ascend_store_connector.py]
                      └─ get_zmq_rpc_path_lookup(vllm_config)
                           │  [vllm-ascend/…/pool_scheduler.py]
                           ├─ reads kv_transfer_config.kv_connector_extra_config
                           │   → "lookup_rpc_port" (preferred)
                           │   → "mooncake_rpc_port" (deprecated fallback + warning)
                           │   → default 0
                           └─ returns "ipc://<VLLM_RPC_BASE_PATH>/lookup_rpc_port_<N>_dp_rank<R>"
                                └─ zmq.REP socket bound at that path
                                     └─ background thread starts (process_request loop)

# ---- SCHEDULER PROCESS ----

vllm serve ...
  └─ Scheduler.__init__()  [vllm/vllm/v1/core/sched/scheduler.py]
       └─ KVConnectorBase_V1 subclass instantiated with role=SCHEDULER
            └─ AscendStoreConnector.__init__(vllm_config, role=SCHEDULER)
                 │   [vllm-ascend/…/ascend_store_connector.py]
                 └─ KVPoolScheduler.__init__(vllm_config, use_layerwise)
                      │  [vllm-ascend/…/pool_scheduler.py]
                      └─ LookupKeyClient.__init__(vllm_config)
                           └─ get_zmq_rpc_path_lookup(vllm_config)
                                └─ (same derivation as above → must match server)
                                     └─ zmq.REQ socket connected to that path
```

> **Key invariant:** Both sides call `get_zmq_rpc_path_lookup(vllm_config)` with the same `vllm_config`. The resulting IPC path must be identical. If two vLLM instances run on the same host, they must use different `lookup_rpc_port` values to avoid socket collisions.

---

### 4.2 Per-request path — lookup query

When a new request arrives and the scheduler needs to decide how many tokens can be loaded from the KV Pool, it triggers the ZMQ RPC round-trip built on the port resolved above.

```
# ---- SCHEDULER PROCESS (per scheduling cycle) ----

Scheduler.schedule()                            [vllm/vllm/v1/core/sched/scheduler.py:348]
  └─ (for each new waiting request)
       └─ connector.get_num_new_matched_tokens(request, num_local_computed_tokens)
            │  [vllm-ascend/…/ascend_store_connector.py:114]
            └─ KVPoolScheduler.get_num_new_matched_tokens(request, num_computed_tokens)
                 │  [vllm-ascend/…/pool_scheduler.py:56]
                 └─ self.client.lookup(token_len, request.block_hashes)
                      │  [vllm-ascend/…/pool_scheduler.py:355]  ← LookupKeyClient.lookup()
                      └─ zmq.REQ.send_multipart([token_len_bytes] + hash_frames)
                           │  (blocks on network call over the IPC socket)
                           │
                           │  ~~~ ZMQ IPC channel (path contains lookup_rpc_port) ~~~
                           │
                           ▼
# ---- WORKER PROCESS (background thread, rank 0) ----

LookupKeyServer.process_request()   [background daemon thread]
  │  [vllm-ascend/…/ascend_store_connector.py:250]
  └─ zmq.REP.recv_multipart()
       └─ pool_worker.lookup_scheduler(token_len, hashes_str, use_layerwise)
            │  [vllm-ascend/…/pool_worker.py:577]
            └─ token_database.process_tokens(token_len, block_hashes)
                 │  → generates block keys
            └─ m_store.exists(keys)           ← queries backend (Mooncake/Memcache/Yuanrong)
            └─ returns: num_hit_tokens (int)
       └─ zmq.REP.send(result.to_bytes(4, "big"))

# ---- back in SCHEDULER PROCESS ----

LookupKeyClient.lookup() ← zmq.REQ.recv() returns num_hit_tokens
  └─ KVPoolScheduler.get_num_new_matched_tokens()
       └─ computes need_to_allocate = num_hit_tokens - num_local_computed_tokens
            └─ returns (need_to_allocate, load_async)
                 └─ Scheduler.schedule() uses this to decide block allocation
```

---

### 4.3 Complete function list in call order

| Step | Process | Function | File |
|------|---------|----------|------|
| 1 | Both | `AscendStoreConnector.__init__` | `ascend_store_connector.py` |
| 2 | Worker rank 0 | `LookupKeyServer.__init__` | `ascend_store_connector.py` |
| 3 | Both | `get_zmq_rpc_path_lookup` | `pool_scheduler.py` |
| 4 | Worker rank 0 | `make_zmq_socket` (bind, REP) | vllm `network_utils` |
| 5 | Scheduler | `KVPoolScheduler.__init__` → `LookupKeyClient.__init__` | `pool_scheduler.py` |
| 6 | Scheduler | `make_zmq_socket` (connect, REQ) | vllm `network_utils` |
| 7 | Scheduler | `Scheduler.schedule` | vllm `scheduler.py` |
| 8 | Scheduler | `AscendStoreConnector.get_num_new_matched_tokens` | `ascend_store_connector.py` |
| 9 | Scheduler | `KVPoolScheduler.get_num_new_matched_tokens` | `pool_scheduler.py` |
| 10 | Scheduler | `LookupKeyClient.lookup` | `pool_scheduler.py` |
| 11 | Worker (thread) | `LookupKeyServer.process_request` (loop) | `ascend_store_connector.py` |
| 12 | Worker (thread) | `KVPoolWorker.lookup_scheduler` | `pool_worker.py` |
| 13 | Worker (thread) | `KVPoolWorker.token_database.process_tokens` + `m_store.exists` | `pool_worker.py` |

---

## 5) Why this key exists — history from upstream issues/PRs

### Primary design evolution

- **RFC #4329** proposed refactoring Mooncake-specific pooling into a backend-extensible Ascend connector architecture.
- **PR #4438** (`[Feature][main] reconstruction kvpool connector to ascend connector`) implemented that shift:
  - renamed MooncakeStoreConnector to AscendStoreConnector
  - introduced backend abstraction
  - migrated docs/config examples from `mooncake_rpc_port` to `lookup_rpc_port`
  - kept compatibility fallback in code (`lookup_rpc_port` preferred, `mooncake_rpc_port` fallback)

### Later continuity

- **PR #5719** moved/refactored distributed KV-transfer code layout; current `pool_scheduler.py` path retained same lookup port logic.
- **PR #7434** revised KV Pool docs and reinforced `lookup_rpc_port` in user guide.
- **PR #7825** updated PD-colocated Mooncake tutorial and continued usage aligned with AscendStoreConnector naming/configs.

---

## 5) Practical usage guidance

For KV Pool / AscendStoreConnector configs, set:

- `kv_connector_extra_config.lookup_rpc_port` to a unique value per instance.

Example shape:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lookup_rpc_port": "1",
    "backend": "mooncake"
  }
}
```

Notes:

- `rpc_lookup_port` is not the code/documented key; use `lookup_rpc_port`.
- `mooncake_rpc_port` may still work for now, but is explicitly deprecated in code warning path.

---

## 6) Source map used for this summary

- Local submodule code/docs:
  - `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
  - `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
  - `vllm-ascend/docs/source/user_guide/feature_guide/kv_pool.md`
  - `vllm/vllm/config/kv_transfer.py` + repo-wide search in `vllm` (no matching key)
- Upstream issue/PR context:
  - RFC issue: `vllm-project/vllm-ascend#4329`
  - PRs: `#4438`, `#5719`, `#7434`, `#7825`, plus `#6229` context where examples use `lookup_rpc_port`.
