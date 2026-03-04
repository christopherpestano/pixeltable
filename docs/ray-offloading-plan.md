# Plan: Ray Cluster Offloading for Pixeltable UDFs

## Context

Pixeltable UDFs currently execute locally via asyncio concurrency. For compute-heavy inference workloads, users need the ability to offload UDF execution to a Ray cluster. This should be **optional** — when no Ray cluster is configured, execution continues locally with zero overhead. The design leverages the existing `resource_pool` + `Scheduler` abstraction with one targeted fix to the evaluator dispatch.

## Design

UDFs opt into Ray offloading by setting `resource_pool='ray'`. A new `RayScheduler` (subclass of `Scheduler`) handles execution. When a Ray cluster is configured (via config/env vars), calls are submitted as `ray.remote` tasks. When Ray is unavailable, the scheduler falls back to local execution transparently.

### Integration Point

The existing scheduler dispatch in `FnCallEvaluator.schedule()` (evaluators.py:132-145) routes calls to `self.dispatcher.schedulers[resource_pool]` when `resource_pool` is set — but **only for batched and async UDF paths**. The sync non-batched path (lines 152-155) has no `resource_pool` check and silently ignores the scheduler. This must be fixed.

The `_init_schedulers()` method (expr_eval_node.py:195-208) iterates `SCHEDULERS` and instantiates the first one whose `matches()` returns `True`. Adding `RayScheduler` to the `SCHEDULERS` list enables scheduler creation. The `matches()` classmethod is static and does not import Ray, so it works even when Ray is not installed.

## File Changes

### 1. `pixeltable/config.py` — Add `ray` config section

Add to `KNOWN_CONFIG_OPTIONS` (after line 218):

```python
'ray': {
    'address': 'Ray cluster address (e.g., "auto", "ray://host:10001")',
    'namespace': 'Ray namespace (optional)',
    'num_cpus': 'CPUs per remote task (optional, default: 1)',
    'num_gpus': 'GPUs per remote task (optional, default: 0)',
    'runtime_env': 'JSON-serialized Ray runtime_env dict (optional)',
},
```

This enables config via `~/.pixeltable/config.toml`:
```toml
[ray]
address = "ray://my-cluster:10001"
num_gpus = 1
```

Or via environment variables: `RAY_ADDRESS`, `RAY_NAMESPACE`, `RAY_NUM_CPUS`, `RAY_NUM_GPUS`.

Or programmatically: `pixeltable.init({'ray.address': 'auto'})`.

### 2. `pixeltable/env.py` — Add Ray client management

Add to `Env.__init__`:
```python
self._ray_initialized: bool = False
self._ray_module: Any | None = None
```

Add method `get_ray_module() -> Any | None`:
- Returns cached `_ray_module` if already initialized
- Attempts `import ray`; returns `None` if `ImportError`
- Reads `ray.address` from Config; returns `None` if not set
- Calls `ray.init(address=..., namespace=..., runtime_env=..., ignore_reinit_error=True)`
- On failure, logs warning and returns `None`
- Caches result for subsequent calls

### 3. `pixeltable/exec/expr_eval/evaluators.py` — Fix sync UDF scheduler dispatch

**Bug**: Lines 152-155 in `FnCallEvaluator.schedule()` bypass the scheduler for sync non-batched UDFs. The `else` branch must check `resource_pool` before falling through to local execution.

Change the sync path (lines 152-155) from:
```python
else:
    # create a single task for all rows
    task = asyncio.create_task(self.eval(rows_call_args))
    self.dispatcher.register_task(task)
```

To:
```python
else:
    if self.fn_call.resource_pool is not None:
        scheduler = self.dispatcher.schedulers[self.fn_call.resource_pool]
        for item in rows_call_args:
            scheduler.submit(item, self.eval_ctx)
    else:
        # create a single task for all rows
        task = asyncio.create_task(self.eval(rows_call_args))
        self.dispatcher.register_task(task)
```

This mirrors the pattern used in the async branch (lines 141-150) and ensures all three UDF types (batched, async, sync) respect the `resource_pool` setting.

### 4. `pixeltable/exec/expr_eval/schedulers.py` — Add `RayScheduler`

New class `RayScheduler(Scheduler)` following the pattern of `RequestRateScheduler`:

```python
class RayScheduler(Scheduler):
    """Scheduler that offloads UDF execution to a Ray cluster.
    Falls back to local execution when Ray is unavailable."""

    @classmethod
    def matches(cls, resource_pool: str) -> bool:
        return resource_pool == 'ray' or resource_pool.startswith('ray:')
```

Key implementation details:

- **`__init__`**: Gets Ray module via `Env.get().get_ray_module()` (lazy import, safe if Ray not installed). Starts `_main_loop` task. Initializes `_remote_fn_cache: dict[Callable, Any]` for caching `ray.remote`-wrapped functions.

- **`_get_remote_fn(py_fn)`**: Lazily wraps `py_fn` with `ray.remote(num_cpus=..., num_gpus=...)` using values from Config. Caches the result. Both sync and async functions are supported directly — Ray 2.x handles both natively via `@ray.remote`.

- **`_main_loop()`**: Reads from priority queue, dispatches to `_exec()` as tasks.

- **`_exec(request, exec_ctx, num_retries)`**: Core execution:
  - If `self.ray is None`: falls back to local execution (calls `pxt_fn.exec`/`aexec`/`exec_batch`/`aexec_batch` directly, matching the patterns in `FnCallEvaluator`)
  - If batched: submits one `ray.remote` task per batch via `remote_fn.remote(*batch_args, **batch_kwargs)`, awaits result via `await obj_ref` (Ray 2.x ObjectRef is directly awaitable)
  - If scalar: submits one `ray.remote` task per row, awaits result via `await obj_ref`
  - On success: sets `row[slot_idx] = result`, calls `dispatcher.dispatch(rows, exec_ctx)`
  - On exception: sets `row.set_exc(slot_idx, exc)`, calls `dispatcher.dispatch_exc(...)`

- **Sync UDFs via Ray**: With the evaluators.py fix above, sync non-batched UDFs with `resource_pool='ray'` now route through the scheduler instead of running sequentially in a single asyncio task — gaining true parallelism across cluster workers.

Update `SCHEDULERS` list (line 423):
```python
SCHEDULERS = [RateLimitsScheduler, RequestRateScheduler, RayScheduler]
```

Update `__all__` (line 20):
```python
__all__ = ['RateLimitsScheduler', 'RequestRateScheduler', 'RayScheduler']
```

### 5. `pyproject.toml` — Optional Ray dependency

Add a new dependency group:
```toml
[dependency-groups]
ray = ["ray[default]>=2.0"]
```

Ray remains fully optional — not added to core or dev dependencies.

## UDF Usage

Opt-in at decoration time:
```python
@pxt.udf(resource_pool='ray')
def my_inference(image: pxt.Image) -> dict:
    return run_model(image)
```

Or via the instance decorator pattern (same as existing providers):
```python
@my_fn.resource_pool
def _(model: str) -> str:
    return 'ray'
```

Batched UDFs work naturally:
```python
@pxt.udf(batch_size=32, resource_pool='ray')
def batch_embed(texts: Batch[str]) -> Batch[list[float]]:
    return model.encode(texts)
```

When `RAY_ADDRESS` is not configured, all calls execute locally with no user-visible difference.

## Serialization

- **Module UDFs** (defined in importable modules): Ray workers receive functions by reference via cloudpickle — works out of the box.
- **Notebook/lambda UDFs** (cloudpickle-stored): Transported to workers via cloudpickle — works if the worker environment has matching packages (controlled via `ray.runtime_env` config).
- **Async UDFs**: Supported directly by `ray.remote` in Ray 2.x — no wrapper shim needed. Ray runs async remote functions in an event loop on the worker.

## Error Handling & Fallback

| Scenario | Behavior |
|----------|----------|
| Ray not installed | `get_ray_module()` catches `ImportError`, returns `None`, local fallback |
| `ray.address` not configured | `get_ray_module()` returns `None`, local fallback |
| `ray.init()` connection fails | Logs warning, `get_ray_module()` returns `None`, local fallback |
| Individual Ray task fails | Exception propagated via `row.set_exc()` + `dispatch_exc()` (matches existing scheduler behavior) |

**Important**: `RayScheduler.matches()` is a classmethod with no Ray import — it always succeeds regardless of whether Ray is installed. The actual Ray import is deferred to `__init__` → `Env.get().get_ray_module()`. If Ray is unavailable, `self.ray` is `None` and all tasks fall through to `_exec_local()`.

No automatic retry in the initial implementation. Can be added later following `RequestRateScheduler`'s retry pattern.

## Review Notes — Issues Found and Fixed

1. **Sync UDFs bypassed scheduler** (CRITICAL): `FnCallEvaluator.schedule()` lines 152-155 had no `resource_pool` check in the sync non-batched path. `resource_pool='ray'` would be silently ignored for sync UDFs. Fixed by adding the scheduler dispatch to the `else` branch in evaluators.py.

2. **Wrong Ray async API**: Original plan used `asyncio.wrap_future(obj_ref.future())`. Ray `ObjectRef` has no `.future()` method. In Ray 2.x, `ObjectRef` is directly awaitable: `result = await obj_ref`. Fixed.

3. **Unnecessary async wrapper**: Original plan wrapped async UDFs in `asyncio.run()` shim before `ray.remote()`. Ray 2.x natively supports `@ray.remote` on async functions. Removed the unnecessary wrapper.

## Testing

### New file: `tests/test_ray.py`

**Unit tests (no Ray cluster needed, run in `make test`):**
- `test_ray_scheduler_matches()`: Verify `matches('ray')`, `matches('ray:gpu')`, `not matches('rate-limits:openai')`
- `test_local_fallback_no_config()`: UDF with `resource_pool='ray'` runs locally and produces correct results when `RAY_ADDRESS` is not set
- `test_local_fallback_scalar()`, `test_local_fallback_batched()`, `test_local_fallback_async()`: End-to-end tests creating tables with Ray-pool UDFs, verifying correct computation via local fallback
- `test_ray_config_env_var()`: Verify `RAY_ADDRESS` env var is picked up by Config

**Integration tests (mark with `@pytest.mark.expensive`):**
- `test_scalar_ray_offload()`: With local Ray (`ray.init(num_cpus=2)`), verify scalar UDF runs on Ray
- `test_batched_ray_offload()`: Verify batched UDF runs as single Ray task per batch
- `test_async_udf_ray_offload()`: Verify async UDF works correctly on Ray

### Verification Steps
1. `make format` — format new code
2. `make check` — pass mypy + ruff
3. `make test` — all local-fallback tests pass without Ray installed
4. Manual test with `ray.init()` to verify actual offloading
