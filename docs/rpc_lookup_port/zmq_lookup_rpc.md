# ZMQ in KV Pool Lookup RPC — Deep Dive

## 1. What is ZMQ here?

ZMQ (ZeroMQ) is a high-performance asynchronous messaging library. In the vLLM-Ascend KV Pool,
it is used as a **lightweight local RPC channel** between two in-process roles that run in different
OS processes (or threads) on the same host:

| Role | ZMQ socket type | Description |
|------|----------------|-------------|
| `LookupKeyServer` (worker rank 0) | `zmq.REP` (Reply) | Binds the IPC endpoint; loops waiting for queries |
| `LookupKeyClient` (scheduler) | `zmq.REQ` (Request) | Connects to that endpoint; sends one query at a time |

The channel uses the **IPC transport** (`ipc://`), meaning all communication happens through a
Unix-domain socket on the local filesystem — no TCP overhead, no network involved.

---

## 2. Why ZMQ (and not shared memory / pipes / gRPC)?

- ZMQ `REQ/REP` gives a clean synchronous request–reply semantic: the scheduler **blocks** until
  the worker answers, which is the desired behavior for KV cache lookup during scheduling.
- IPC sockets are significantly faster than TCP sockets for local processes.
- vLLM already depends on ZMQ (used for its own engine RPC bus), so no new dependency is added.
- The socket path encodes `lookup_rpc_port` and `dp_rank`, making it trivially safe to run
  multiple instances on the same host without collision.

---

## 3. Socket address derivation

Both sides independently call the same helper to produce the IPC path:

```python
# pool_scheduler.py — get_zmq_rpc_path_lookup()
def get_zmq_rpc_path_lookup(vllm_config) -> str:
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config
    rpc_port = 0                                         # default
    if "lookup_rpc_port" in extra:
        rpc_port = extra["lookup_rpc_port"]              # preferred key
    elif "mooncake_rpc_port" in extra:
        warnings.warn("mooncake_rpc_port is deprecated, use lookup_rpc_port")
        rpc_port = extra["mooncake_rpc_port"]            # legacy fallback

    dp_rank = get_dp_rank()                              # data-parallel rank

    base = os.environ.get("VLLM_RPC_BASE_PATH", "/tmp/vllm")
    return f"ipc://{base}/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}"
```

**Key invariant:** both processes call this with the same `vllm_config`, so the returned path is
identical. That is the only "rendezvous" needed.

---

## 4. Startup path — socket creation

### 4.1 Worker side (`LookupKeyServer.__init__`)

```
vllm serve ...
  └─ EngineCore.__init__()                         [vllm/v1/engine/core.py]
       └─ KVConnectorBase_V1 subclass (role=WORKER)
            └─ AscendStoreConnector.__init__(vllm_config, role=WORKER)
                 │                             [ascend_store_connector.py]
                 ├─ KVPoolWorker.__init__(vllm_config, use_layerwise)
                 │    └─ initialises token_database, m_store, etc.
                 │
                 └─ LookupKeyServer.__init__(pool_worker, vllm_config, use_layerwise)
                      │
                      ├─ path = get_zmq_rpc_path_lookup(vllm_config)
                      │         → "ipc:///tmp/vllm/lookup_rpc_port_1_dp_rank0"
                      │
                      ├─ self.socket = make_zmq_socket(zmq.REP, path, bind=True)
                      │   # vllm/utils/network_utils.py
                      │   # Creates a zmq.Context(), opens REP socket, binds to path.
                      │
                      └─ threading.Thread(target=self.process_request, daemon=True).start()
                           # background thread spins forever waiting for queries
```

### 4.2 Scheduler side (`LookupKeyClient.__init__`)

```
vllm serve ...
  └─ Scheduler.__init__()                          [vllm/v1/core/sched/scheduler.py]
       └─ KVConnectorBase_V1 subclass (role=SCHEDULER)
            └─ AscendStoreConnector.__init__(vllm_config, role=SCHEDULER)
                 └─ KVPoolScheduler.__init__(vllm_config, use_layerwise)
                      │                        [pool_scheduler.py]
                      └─ LookupKeyClient.__init__(vllm_config)
                           │
                           ├─ path = get_zmq_rpc_path_lookup(vllm_config)
                           │         → "ipc:///tmp/vllm/lookup_rpc_port_1_dp_rank0"
                           │         (must match server's path — same config)
                           │
                           └─ self.socket = make_zmq_socket(zmq.REQ, path, bind=False)
                               # Creates zmq.Context(), opens REQ socket, connects to path.
```

At this point the IPC channel is live. The server thread is blocked on `recv_multipart()`.

---

## 5. Per-request path — lookup query

Every scheduling cycle, for each new waiting request, the scheduler triggers a synchronous
round-trip over the ZMQ channel:

```
# ─── SCHEDULER PROCESS ─────────────────────────────────────────────────────────

Scheduler.schedule()                        [vllm/v1/core/sched/scheduler.py:348]
  └─ for each request in waiting_queue:
       └─ connector.get_num_new_matched_tokens(request, num_local_computed_tokens)
            │                                  [ascend_store_connector.py:114]
            └─ KVPoolScheduler.get_num_new_matched_tokens(request, num_computed_tokens)
                 │                             [pool_scheduler.py:56]
                 ├─ token_len  = len(request.prompt_token_ids) - num_computed_tokens
                 ├─ hash_frames = [h.to_bytes() for h in request.block_hashes]
                 │
                 └─ self.client.lookup(token_len, request.block_hashes)
                      │               [pool_scheduler.py:355]  ← LookupKeyClient.lookup()
                      │
                      │  # Serialise the query
                      ├─ token_len_bytes = token_len.to_bytes(4, "big")
                      ├─ frames = [token_len_bytes] + hash_frames
                      │
                      ├─ zmq.REQ.send_multipart(frames)
                      │   ┌───────────────────────────────────────┐
                      │   │  ZMQ IPC channel                      │
                      │   │  ipc://.../lookup_rpc_port_1_dp_rank0 │
                      │   └───────────────────────────────────────┘
                      │                        ↓↓↓ (blocks here) ↓↓↓
                      └─ num_hit_tokens = int.from_bytes(zmq.REQ.recv(), "big")


# ─── WORKER PROCESS (background daemon thread) ──────────────────────────────────

LookupKeyServer.process_request()           [ascend_store_connector.py:250]
  └─ loop forever:
       ├─ frames = zmq.REP.recv_multipart()
       │   # frames[0] = token_len (4 bytes), frames[1..] = block hashes
       │
       ├─ token_len  = int.from_bytes(frames[0], "big")
       ├─ hashes_str = [f.decode() for f in frames[1:]]
       │
       ├─ result = pool_worker.lookup_scheduler(token_len, hashes_str, use_layerwise)
       │    │      [pool_worker.py:577]
       │    ├─ block_keys = token_database.process_tokens(token_len, block_hashes)
       │    │               # converts token hashes → backend storage keys
       │    └─ num_hit = m_store.exists(block_keys)
       │                 # queries actual backend (Mooncake / Memcache / Yuanrong)
       │                 # returns count of keys that are present in the KV store
       │
       └─ zmq.REP.send(result.to_bytes(4, "big"))
            # ↑ unblocks the scheduler's recv() above


# ─── BACK IN SCHEDULER PROCESS ──────────────────────────────────────────────────

KVPoolScheduler.get_num_new_matched_tokens()
  └─ num_hit_tokens received
       └─ need_to_allocate = num_hit_tokens - num_local_computed_tokens
            └─ return (need_to_allocate, load_async=True)
                 └─ Scheduler.schedule() uses this to decide block allocation
```

---

## 6. Message wire format

| Frame index | Content | Encoding |
|-------------|---------|----------|
| 0 | `token_len` | 4-byte big-endian unsigned int |
| 1 … N | `block_hashes[i]` | one ZMQ frame per block hash |

Reply is a single frame: 4-byte big-endian unsigned int = `num_hit_tokens`.

This is an intentionally minimal encoding — no protobuf, no JSON — to keep the round-trip as fast
as possible.

---

## 7. Multi-instance isolation

If two vLLM instances are co-located on the same machine, they must set **different**
`lookup_rpc_port` values:

```json
// Instance A
{ "kv_connector_extra_config": { "lookup_rpc_port": "1", "backend": "mooncake" } }

// Instance B
{ "kv_connector_extra_config": { "lookup_rpc_port": "2", "backend": "mooncake" } }
```

This produces distinct IPC paths:
- `ipc:///tmp/vllm/lookup_rpc_port_1_dp_rank0`
- `ipc:///tmp/vllm/lookup_rpc_port_2_dp_rank0`

Without this, the second instance's `REQ` socket would connect to the first instance's `REP`
socket, causing cross-instance KV lookup corruption.

---

## 8. Summary of involved source files

| File | Role |
|------|------|
| `vllm-ascend/…/pool_scheduler.py` | `get_zmq_rpc_path_lookup`, `LookupKeyClient`, `KVPoolScheduler` |
| `vllm-ascend/…/ascend_store_connector.py` | `LookupKeyServer`, `AscendStoreConnector` |
| `vllm-ascend/…/pool_worker.py` | `KVPoolWorker.lookup_scheduler`, `m_store.exists` |
| `vllm/vllm/utils/network_utils.py` | `make_zmq_socket` (generic socket factory) |
| `vllm/vllm/v1/engine/core.py` | `EngineCore` (instantiates WORKER connector) |
| `vllm/vllm/v1/core/sched/scheduler.py` | `Scheduler` (instantiates SCHEDULER connector, drives `schedule()`) |
