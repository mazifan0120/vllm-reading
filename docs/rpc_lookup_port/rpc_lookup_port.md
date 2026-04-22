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

## 4) Why this key exists (history from upstream issues/PRs)

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
