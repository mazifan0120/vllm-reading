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

## 6) Practical usage guidance

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

## 7) What happens when two instances share the same `lookup_rpc_port`?

### ZMQ IPC is backed by a Unix domain socket file

When ZMQ binds an `ipc://` address it creates a socket file on the filesystem, e.g.:

```
/tmp/vllm/lookup_rpc_port_1_dp_rank0   ← socket inode
```

Both sides independently call `get_zmq_rpc_path_lookup(vllm_config)` to derive this path. If two
vLLM instances produce the same path string, they will compete for the same socket file.

### Two failure modes

#### Mode A — Second `REP` bind fails hard (if socket file already exists)

When instance B's `LookupKeyServer` tries to call `socket.bind(path)` while instance A is still
running and holds the file open:

```
zmq.error.ZMQError: Address already in use
  File "pool_scheduler.py", line ...
    make_zmq_socket(zmq.REP, path, bind=True)
```

Instance B crashes at startup before serving any request. This is the *safe* failure: noisy but
self-limiting.

#### Mode B — Second `REQ` connects to the wrong `REP` (silent data corruption)

If the socket file is present but instance A's `REP` socket is **no longer listening** (e.g., A
crashed without cleaning up the file), instance B's `LookupKeyClient` (`zmq.REQ`) still connects
successfully to that stale path. Two sub-cases:

```
# Sub-case B1 — nobody listening
Scheduler B calls LookupKeyClient.lookup(token_len, block_hashes)
  └─ REQ.send_multipart(frames)
       └─ REQ.recv() blocks forever    ← scheduler hangs; all requests stall
```

```
# Sub-case B2 — instance A partially alive
Instance B scheduler  ──REQ.send──►  Instance A LookupKeyServer
                      ◄──REP.recv──  returns hit count from A's m_store (wrong data)
```

In sub-case B2, instance B's scheduling decisions are based on **instance A's KV cache state**:

- B may skip prefill for tokens that are **not** in B's own KV store (falsely treated as cached).
- B may issue redundant prefill for tokens that **are** in B's KV store but unknown to A.
- No exception is raised; the model output remains syntactically correct, making this bug hard to detect.

### Summary table

| Scenario | What ZMQ does | Symptom |
|----------|--------------|---------|
| Both instances running, same port | Second bind raises `EADDRINUSE` | Instance B crashes at startup |
| Instance A dead, stale socket file, nobody listening | B's REQ connects, then blocks on recv | Scheduler B hangs indefinitely |
| Instance A dead, stale socket file, A briefly restarts | B's REQ talks to A's REP | Silent cross-instance KV lookup corruption |

### Mitigation on bare metal / same host

Set a different `lookup_rpc_port` per instance **or** set `VLLM_RPC_BASE_PATH` to a per-instance
directory:

```bash
# Instance A
VLLM_RPC_BASE_PATH=/tmp/vllm_A \
  vllm serve ... --kv-connector-extra-config '{"lookup_rpc_port": "1"}'

# Instance B — different base path; same port value is now safe
VLLM_RPC_BASE_PATH=/tmp/vllm_B \
  vllm serve ... --kv-connector-extra-config '{"lookup_rpc_port": "1"}'
```

---

## 8) Docker: are two containers with the same `lookup_rpc_port` isolated?

### Short answer: **Yes, by default** — but with important caveats.

### Why they are isolated by default

Docker (via Linux namespaces) gives each container its own **mount namespace**. The container's
filesystem tree — including `/tmp` — is completely independent of the host's `/tmp` and of other
containers' `/tmp`:

```
Host filesystem:   /tmp/vllm/...     (separate inode namespace)
Container A:       /tmp/vllm/...     (overlay on container A's rootfs layer)
Container B:       /tmp/vllm/...     (overlay on container B's rootfs layer)
```

A Unix domain socket is just a file. Because the two containers' filesystems are **separate**,
there is **no sharing** of the socket inode:

- Container A: `bind("ipc:///tmp/vllm/lookup_rpc_port_1_dp_rank0")` → socket in A's `/tmp`
- Container B: `bind("ipc:///tmp/vllm/lookup_rpc_port_1_dp_rank0")` → socket in B's `/tmp`

These are different filesystem objects. Neither bind fails, and neither client can cross-connect.

### When Docker does **NOT** isolate you (caveats)

#### Caveat 1 — Shared host volume for `/tmp`

```bash
docker run -v /tmp:/tmp vllm-instance-A ...
docker run -v /tmp:/tmp vllm-instance-B ...
```

Both containers see the same `/tmp` on the host. The socket file collision is identical to the
bare-metal case. **Fix**: use per-instance volumes or set `VLLM_RPC_BASE_PATH` to a unique path.

#### Caveat 2 — `--network=host` is **irrelevant** for IPC sockets

`--network=host` shares the network namespace (TCP/UDP ports), **not** the mount namespace. IPC
(`ipc://`) sockets are filesystem-based and remain isolated regardless of `--network` mode.

#### Caveat 3 — `--ipc=host` is also **irrelevant**

Docker's `--ipc` flag controls the **System V IPC namespace** (shared memory, semaphores, message
queues). ZMQ IPC sockets are plain Unix domain socket files — not System V IPC — so `--ipc=host`
does not expose any collision risk.

#### Caveat 4 — Kubernetes shared `emptyDir` on `/tmp`

If two pods on the same node mount the same `emptyDir` or host-path volume at `/tmp`, the
isolation collapses to the bare-metal case. Use unique `lookup_rpc_port` values or unique
`VLLM_RPC_BASE_PATH` paths in that configuration.

### Decision matrix

| Docker / K8s configuration | Isolated? | Risk |
|---------------------------|-----------|------|
| Default (no special flags) | ✅ Yes | None |
| `-v /tmp:/tmp` (shared host `/tmp`) | ❌ No | Same as bare-metal collision |
| `--tmpfs /tmp` (per-container tmpfs) | ✅ Yes | None |
| `--network=host` | ✅ Yes | No effect on IPC sockets |
| `--ipc=host` | ✅ Yes | System V IPC only; ZMQ unaffected |
| `--ipc=container:<A>` | ✅ Yes | Same reason |
| Kubernetes shared `emptyDir` on `/tmp` | ❌ No | Treat as shared volume |

### How to be safe across all environments

| Environment | Safe configuration |
|-------------|-------------------|
| Bare metal / VM | Different `lookup_rpc_port` **or** different `VLLM_RPC_BASE_PATH` per instance |
| Docker (default) | No action needed; filesystem is already isolated |
| Docker with `-v /tmp:/tmp` | Different `lookup_rpc_port` **or** different `VLLM_RPC_BASE_PATH` |
| Kubernetes (shared host-path `/tmp`) | Different `lookup_rpc_port` per instance |

**Universal rule:** always set `lookup_rpc_port` to a value unique per vLLM instance, regardless
of deployment topology. The cost is a single config key; the benefit is guaranteed safety.

---

## 9) Source map used for this summary

- Local submodule code/docs:
  - `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
  - `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
  - `vllm-ascend/docs/source/user_guide/feature_guide/kv_pool.md`
  - `vllm/vllm/config/kv_transfer.py` + repo-wide search in `vllm` (no matching key)
- Upstream issue/PR context:
  - RFC issue: `vllm-project/vllm-ascend#4329`
  - PRs: `#4438`, `#5719`, `#7434`, `#7825`, plus `#6229` context where examples use `lookup_rpc_port`.
