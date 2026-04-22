# `lookup_rpc_port` Collision and Docker Isolation

## Background

As established in `zmq_lookup_rpc.md`, the ZMQ IPC socket path used for KV-cache lookup is:

```
ipc://<VLLM_RPC_BASE_PATH>/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}
```

This path defaults to something like `/tmp/vllm/lookup_rpc_port_1_dp_rank0`.

Both the worker (`LookupKeyServer`, `zmq.REP`, binds) and the scheduler (`LookupKeyClient`,
`zmq.REQ`, connects) are derived from the **same config value**, so they independently compute the
same string and rendezvous on it.

---

## Part 1 — What happens when two instances on the **same host** share the same `lookup_rpc_port`?

### ZMQ IPC is backed by a Unix domain socket file

When ZMQ binds an `ipc://` address it creates a socket file on the filesystem, e.g.:

```
/tmp/vllm/lookup_rpc_port_1_dp_rank0   ← socket inode
```

### Two failure modes

#### Mode A — Second `REP` bind fails hard (if socket file already exists)

When instance B's `LookupKeyServer` tries to call `socket.bind(path)`:

```
zmq.error.ZMQError: Address already in use
  File "pool_scheduler.py", line ...
    make_zmq_socket(zmq.REP, path, bind=True)
```

This raises an exception during startup, crashing the worker process before any requests are
served. This is the *safe* failure: it is noisy but self-limiting.

Whether this happens depends on the ZMQ version and whether the previous socket file was cleaned
up. If instance A is still running, its socket file is held open and mode A typically fires.

#### Mode B — Second `REQ` connects to the wrong `REP` (silent data corruption)

If the socket file exists but instance A's REP socket is **no longer listening** (e.g., instance A
crashed but did not clean up the file), instance B's `LookupKeyClient` (`zmq.REQ`) will still
connect successfully to that stale path.

What happens next:

```
Scheduler B calls LookupKeyClient.lookup(token_len, block_hashes)
  └─ REQ.send_multipart(frames)
       │
       │  IPC path → /tmp/vllm/lookup_rpc_port_1_dp_rank0
       │  ... but nobody is listening on REP side!
       │
       └─ REQ.recv() blocks forever    ← scheduler hangs, all requests stall
```

If by bad luck instance A is still alive but partially degraded, instance B's scheduler queries
instance A's KV worker:

```
Instance B scheduler  ──REQ.send──►  Instance A LookupKeyServer
                      ◄──REP.recv──  Instance A pool_worker.lookup_scheduler()
                                      returns cache hit count from A's m_store
```

Instance B's scheduling decisions are now based on cache state that belongs to **instance A** —
completely wrong answers. The result is:

- B may skip prefill for tokens that are **not** in B's own KV store (treats A's hits as its own).
- B may issue redundant prefill for tokens that **are** in B's KV store but A doesn't know about.
- No explicit error is raised; the model output is still *syntactically* correct (the LLM just runs
  extra or fewer prefill steps), so the bug is hard to detect.

### Summary table

| Scenario | What ZMQ does | Symptom |
|----------|--------------|---------|
| Both instances running, same port | Second bind raises `EADDRINUSE` | Instance B crashes at startup |
| Instance A dead, stale socket file, nobody listening | B's REQ connects, then blocks on recv | Scheduler B hangs indefinitely |
| Instance A dead, stale socket file, A briefly restarts | B's REQ talks to A's REP | Silent cross-instance KV lookup corruption |

### Mitigation on bare metal / same host

Set different `lookup_rpc_port` per instance **or** set `VLLM_RPC_BASE_PATH` to a per-instance
directory:

```bash
# Instance A
VLLM_RPC_BASE_PATH=/tmp/vllm_A \
  vllm serve ... --kv-connector-extra-config '{"lookup_rpc_port": "1"}'

# Instance B — different base path, same port value is now safe
VLLM_RPC_BASE_PATH=/tmp/vllm_B \
  vllm serve ... --kv-connector-extra-config '{"lookup_rpc_port": "1"}'
```

---

## Part 2 — Docker: are two containers with the **same** `lookup_rpc_port` isolated?

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

A Unix domain socket (IPC) is just a file. Because the two containers' filesystems are
**separate**, there is **no sharing** of the socket inode. Container A's `REP` bind and container
B's `REP` bind each create their own socket file in their own namespaces — no collision.

Concretely:
- Container A: `bind("ipc:///tmp/vllm/lookup_rpc_port_1_dp_rank0")` → creates socket in A's `/tmp`
- Container B: `bind("ipc:///tmp/vllm/lookup_rpc_port_1_dp_rank0")` → creates socket in B's `/tmp`
- These are different filesystem objects. Neither bind fails. Neither client can cross-connect.

### When Docker does **NOT** isolate you (caveats)

#### Caveat 1 — Shared host volume for `/tmp`

```bash
docker run -v /tmp:/tmp vllm-instance-A ...
docker run -v /tmp:/tmp vllm-instance-B ...
```

Both containers now see the same `/tmp` on the host. The socket file collision is identical to the
bare-metal case: second bind fails (`EADDRINUSE`) or clients cross-connect.

**Fix**: mount per-instance volumes or set `VLLM_RPC_BASE_PATH` to a unique path inside each
container.

#### Caveat 2 — Shared `/tmp` via `--tmpfs` or bind-mount between containers

Same consequence as caveat 1.

#### Caveat 3 — `--pid=host` (shared PID namespace) **does not** help or hurt IPC

Sharing the PID namespace does not share the filesystem; sockets are still isolated.

#### Caveat 4 — `--network=host` is **irrelevant** for IPC sockets

`--network=host` shares the network namespace (TCP/UDP ports), **not** the mount namespace. IPC
(`ipc://`) sockets are filesystem-based and remain isolated regardless of `--network` mode.

#### Caveat 5 — `--ipc=host` or `--ipc=container:<name>`

Docker's `--ipc` flag controls the **System V IPC namespace** (shared memory segments, semaphores,
message queues). ZMQ IPC sockets are **not** System V IPC — they are plain Unix domain socket
files. So `--ipc=host` does **not** expose the socket collision risk either.

### Decision matrix

| Docker configuration | Isolated? | Risk |
|----------------------|-----------|------|
| Default (no special flags) | ✅ Yes | None |
| `-v /tmp:/tmp` (shared host `/tmp`) | ❌ No | Same as bare-metal collision |
| `--tmpfs /tmp` (per-container tmpfs) | ✅ Yes | None |
| `--network=host` | ✅ Yes | No effect on IPC sockets |
| `--ipc=host` | ✅ Yes | System V IPC only; ZMQ unaffected |
| `--ipc=container:<A>` | ✅ Yes | Same reason |
| Kubernetes shared `emptyDir` on `/tmp` | ❌ No | Treat as shared volume |

---

## Part 3 — How to be safe in all environments

| Environment | Safe configuration |
|-------------|-------------------|
| Bare metal / VM | Different `lookup_rpc_port` **or** different `VLLM_RPC_BASE_PATH` per instance |
| Docker (default) | No action needed; filesystem is already isolated |
| Docker with `-v /tmp:/tmp` | Different `lookup_rpc_port` **or** different `VLLM_RPC_BASE_PATH` |
| Kubernetes pod (same node) | Different `lookup_rpc_port` if sharing a host-mounted `/tmp` volume |

The cleanest universal rule: **always set `lookup_rpc_port` to a value unique per vLLM instance**,
regardless of whether you think they are isolated. The cost is a single config key; the benefit is
guaranteed safety across all deployment topologies.
