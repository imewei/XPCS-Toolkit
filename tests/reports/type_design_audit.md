# Type Design Audit — Phase 4

Generated: 2026-02-19
Agent: type-analyst (type design and encapsulation specialist)

---

## Executive Summary

This audit evaluates 22 significant types across 7 source files in the xpcsviewer codebase. Each type is rated on four axes (1-5 scale):

- **Encapsulation**: How well the type hides internals and exposes a clean API
- **Invariant Expression**: How well the type system captures and communicates constraints
- **Usefulness**: How essential and well-designed the type is for its domain
- **Enforcement**: How effectively invalid states are prevented at construction time

Key findings: The schema types (`QMapSchema`, `GeometryMetadata`, `G2Data`) score highest overall, demonstrating strong frozen-dataclass patterns with thorough `__post_init__` validation. The weakest types are `XpcsFile` (god-object with dynamic `__getattr__` and no type-safe interface) and `MaskBase` (hierarchy with untyped `zero_loc` and no shape invariant enforcement).

---

## 1. Rating Table

| Type | File | Encaps. | Invariant | Useful | Enforce | Avg |
|------|------|:-------:|:---------:|:------:|:-------:|:---:|
| **QMapSchema** | `schemas/validators.py:34` | 5 | 5 | 5 | 5 | **5.0** |
| **GeometryMetadata** | `schemas/validators.py:267` | 5 | 4 | 5 | 4 | **4.5** |
| **G2Data** | `schemas/validators.py:403` | 5 | 4 | 5 | 4 | **4.5** |
| **PartitionSchema** | `schemas/validators.py:531` | 5 | 4 | 4 | 4 | **4.3** |
| **MaskSchema** | `schemas/validators.py:686` | 5 | 4 | 4 | 4 | **4.3** |
| **SamplerConfig** | `fitting/results.py:56` | 4 | 4 | 4 | 4 | **4.0** |
| **FitDiagnostics** | `fitting/results.py:101` | 3 | 3 | 4 | 3 | **3.3** |
| **FitResult** | `fitting/results.py:730` | 3 | 3 | 4 | 3 | **3.3** |
| **NLSQResult** | `fitting/results.py:186` | 2 | 2 | 4 | 2 | **2.5** |
| **BackendProtocol** | `backends/_base.py:19` | 5 | 3 | 5 | 3 | **4.0** |
| **JAXBackend** | `backends/_jax_backend.py:38` | 4 | 3 | 5 | 3 | **3.8** |
| **MaskBase** | `simplemask/area_mask.py:67` | 2 | 1 | 3 | 1 | **1.8** |
| **MaskList** | `simplemask/area_mask.py:125` | 2 | 1 | 3 | 1 | **1.8** |
| **MaskThreshold** | `simplemask/area_mask.py:220` | 3 | 2 | 3 | 2 | **2.5** |
| **MaskAssemble** | `simplemask/area_mask.py:337` | 3 | 2 | 4 | 2 | **2.8** |
| **WorkerState** | `threading/async_workers.py:29` | 5 | 5 | 3 | 5 | **4.5** |
| **WorkerResult** | `threading/async_workers.py:41` | 4 | 3 | 3 | 3 | **3.3** |
| **WorkerStats** | `threading/async_workers.py:66` | 4 | 3 | 3 | 3 | **3.3** |
| **WorkerSignals** | `threading/async_workers.py:115` | 3 | 2 | 4 | 2 | **2.8** |
| **BaseAsyncWorker** | `threading/async_workers.py:151` | 3 | 2 | 5 | 2 | **3.0** |
| **WorkerManager** | `threading/async_workers.py:528` | 3 | 2 | 4 | 2 | **2.8** |
| **UnifiedTask** | `threading/unified_threading.py:50` | 3 | 3 | 4 | 3 | **3.3** |
| **UnifiedThreadingManager** | `threading/unified_threading.py:109` | 3 | 2 | 4 | 2 | **2.8** |
| **TaskPriority** | `threading/unified_threading.py:30` | 5 | 5 | 4 | 5 | **4.8** |
| **TaskType** | `threading/unified_threading.py:40` | 5 | 5 | 4 | 5 | **4.8** |
| **XpcsFile** | `xpcs_file.py:196` | 1 | 1 | 5 | 1 | **2.0** |

---

## 2. Detailed Analysis

### 2.1 Schema Types (`xpcsviewer/schemas/validators.py`)

#### QMapSchema (5/5/5/5)

**What it does well:**
- Frozen dataclass prevents mutation after construction
- Defensive copy of all array fields in `__post_init__` prevents aliasing
- Exhaustive validation: shape consistency, 2D dimensionality, dtype (float64), NaN checks, unit literals
- `Literal["nm^-1", "A^-1"]` for units provides compile-time type narrowing
- `to_dict()` returns defensive copies (BUG-047 fix)
- Clean factory methods: `from_dict()` (legacy support) and `from_compute_qmap()` (pipeline output)

**What's weak:**
- `from_compute_qmap()` unit normalization is fragile for non-standard strings (C-016). Uses substring matching (`"A" in q_unit`) which can produce false positives.
- The `type: ignore[arg-type]` comments in `from_compute_qmap()` indicate the unit normalization path can produce strings that don't satisfy the Literal constraint — the type system is bypassed at this point.

**Recommendation:** Replace the string-matching unit normalization with an explicit mapping dict.

#### GeometryMetadata (5/4/5/4)

**What it does well:**
- Frozen dataclass with positive-value validation for `det_dist`, `lambda_`, `pix_dim`
- Shape tuple validated for length and positive dimensions
- `from_dict()` validates NaN and Inf for all critical fields (BUG-048)
- Beam center out-of-bounds produces a warning, not an error — appropriate for real-world data

**What's weak:**
- `bcx`/`bcy` allow negative values without error (only warns when outside bounds). A negative beam center is valid if the beam is off-detector.
- `det_rotation` and `incident_angle` are `float | None` with no range validation. A rotation > 360 or negative angle is physically questionable.
- `from_dict()` calls `tuple(data["shape"])` without validating that `data["shape"]` has exactly 2 elements before passing to the constructor. The constructor validates length, but the error message would be confusing.

**Recommendation:** Add range validation for `det_rotation` (0-360) and `incident_angle` (0-90) in `__post_init__` with warnings (not errors) for out-of-range values.

#### G2Data (5/4/5/4)

**What it does well:**
- Frozen dataclass with defensive array copies
- `q_values` list converted to immutable tuple (BUG-010 fix) — prevents mutation through the frozen dataclass backdoor
- Comprehensive validation: shape consistency, 2D requirement, dtype (float64), NaN in g2/delay_times, non-negative delay_times, strict monotonicity, non-negative errors
- `from_dict()` coerces to float64 (BUG-058 fix)

**What's weak:**
- **NaN in g2 raises ValueError** (C-015): Real XPCS data may have NaN-filled Q-bins. This makes the schema overly strict for production data.
- `to_dict()` returns the internal arrays directly (not copies), unlike `QMapSchema.to_dict()`. Since the dataclass is frozen and arrays were copied on construction, this is technically safe — but external code could still mutate the returned dict's arrays, which wouldn't corrupt the schema but creates inconsistency with QMapSchema's defensive-copy pattern.

**Recommendation:** Either allow NaN in g2 with a per-Q-bin NaN count warning, or document that callers must pre-filter NaN Q-bins before schema construction.

#### PartitionSchema (5/4/4/4)

**What it does well:**
- Frozen dataclass with defensive copies for arrays and immutable tuples for lists
- Validates `num_pts > 0`, list lengths match `num_pts`, non-negative `num_list` values
- Consistency check: warns if `sum(num_list)` differs from `nonzero(partition_map)` by more than `num_pts`
- `from_dict()` handles both `dict` and `GeometryMetadata` for `metadata` field

**What's weak:**
- The `partition_map` dtype check accepts both `int32` and `int64`, but `QMapSchema.mask` requires values of exactly 0 or 1 — `partition_map` has no maximum bin index validation.
- The `Literal["linear", "log"] | None` for `method` is validated in `__post_init__`, which is redundant with the type annotation but necessary at runtime for untyped callers.

#### MaskSchema (5/4/4/4)

**What it does well:**
- Frozen dataclass with defensive array copy
- Validates 2D shape, dtype (bool/int32/int64), binary values (0 or 1)
- Shape consistency warning (not error) between mask and metadata — pragmatic for real data

**What's weak:**
- Shape mismatch between mask and metadata is only a warning, not an error. This could hide bugs where the wrong mask is paired with wrong geometry.

---

### 2.2 Fitting Types (`xpcsviewer/fitting/results.py`)

#### SamplerConfig (4/4/4/4)

**What it does well:**
- Clean dataclass with sensible defaults matching standard NUTS configuration
- `__post_init__` validates all numeric constraints: positive integers, `target_accept_prob` in (0,1)
- Immutable-style usage (no setters, no mutation methods)

**What's weak:**
- `random_seed: int | None` has no type validation — `random_seed=1.5` would pass (C-012 note)
- No upper bound on `num_warmup`, `num_samples`, `num_chains` — astronomically large values could exhaust memory. This is acceptable (user's responsibility) but could be guarded.

#### FitDiagnostics (3/3/4/3)

**What it does well:**
- Validates `divergences >= 0`, `ess_bulk/ess_tail >= 0`
- `converged` property computes convergence from standard thresholds (R-hat < 1.01, ESS > 400, etc.)
- BFMI check included per technical guidelines

**What's weak:**
- **Vacuous truth bug (C-010)**: When `r_hat`, `ess_bulk`, and `ess_tail` dicts are empty, `converged` returns `True` because `any()` on an empty iterable is `False`. Empty diagnostics mean "unknown", not "converged."
- All dict fields use `default_factory=dict`, so a default-constructed `FitDiagnostics()` reports `converged=True` with zero information.
- No type enforcement on dict value types — `r_hat` values could be strings, `ess_bulk` could contain floats instead of ints.

**Recommendation:** Add a non-empty guard: `if not self.r_hat and not self.ess_bulk: return False`. This changes the semantics from "vacuously converged" to "unknown = not converged."

#### NLSQResult (2/2/4/2)

**What it does well:**
- Rich API surface: 8 delegated statistical properties, confidence/prediction intervals, summary text, plotting
- Clean delegation pattern: checks `native_result` first, falls back to `_legacy` fields
- `is_fallback` sentinel flag distinguishes fitting failures from legitimate results
- NaN defaults for legacy fields (BUG-023 fix) — `math.isnan(result.r_squared)` detects failure

**What's weak:**
- **Massive surface area (540 lines)**: 11 property getter/setter pairs with identical boilerplate. Each delegated property repeats the same `if self.native_result is not None and hasattr(...)` pattern. This is a maintenance burden and code smell.
- **`__post_init__` does almost nothing**: Only initializes `_param_names` from `params.keys()`. No validation of `params` values being float, no validation of `chi_squared` range, no validation of `converged` type.
- **health_score/condition_number attribute chains (C-003, C-004)**: `health_score` accesses `self.diagnostics.health_score` — if `diagnostics` returns a `ModelHealthReport` that lacks `.health_score`, `AttributeError` is raised. The `is_healthy` property has a 5-level fallback chain (native, health_score, enum value, enum name, string comparison), but `health_score` itself has no fallback.
- **condition_number** chains `self.diagnostics.identifiability.condition_number` with only a None check on `identifiability`, not on `condition_number`.
- **Hidden scipy dependency** in `get_confidence_interval()` fallback path (C-012).
- **Setters are no-ops when `native_result` is active** — they log warnings but the caller receives no indication that their assignment was silently ignored. This violates the principle of least surprise.

**Recommendation:**
1. Extract the delegation pattern into a descriptor or `__getattr__` override to eliminate boilerplate.
2. Add `hasattr` guard in `health_score` and `condition_number`.
3. Make `params` values validated as `float` in `__post_init__`.
4. Consider making setters raise `AttributeError` instead of silently logging.

Cross-reference: C-003, C-004, C-012 in `contract_audit.md`.

#### FitResult (3/3/4/3)

**What it does well:**
- Non-empty `samples` validation in `__post_init__`
- Shape consistency check: all sample arrays must have same shape
- `to_dict()` produces comprehensive serialization with version metadata and sampler config
- Clean accessor API: `get_mean()`, `get_std()`, `get_hdi()`, `get_samples()` all check param existence

**What's weak:**
- **`predict()` is a placeholder** returning zeros — it requires the model function but doesn't store it. The method exists to satisfy an interface but provides wrong results.
- **`to_dict()` implemented but never called** (C-028). No `from_dict()` deserializer exists. Bayesian results are not persisted across sessions.
- **`samples` values not type-validated** in `__post_init__`. JAX arrays can be stored (C-019) and `.tolist()` works on both, but Python lists would cause `AttributeError`.
- **`diagnostics` defaults to empty `FitDiagnostics()`** which reports `converged=True` (C-010). A newly constructed `FitResult` with empty diagnostics appears converged.
- **`arviz_data`** typed as `az.InferenceData | None` but `az` may not be importable. The type annotation is conditional on arviz availability.

**Recommendation:**
1. Implement `from_dict()` and wire into session persistence.
2. Change `diagnostics` default to require explicit construction.
3. Remove or properly implement `predict()`.

Cross-reference: C-010, C-028 in `contract_audit.md`.

---

### 2.3 Mask Hierarchy (`xpcsviewer/simplemask/area_mask.py`)

#### MaskBase (2/1/3/1)

**What it does well:**
- Simple, understandable base class with clear `evaluate()`/`get_mask()`/`combine_mask()` protocol
- `describe()` provides useful mask statistics

**What's weak:**
- **`zero_loc` is untyped (`np.ndarray | None`)**: No shape constraint (should be `(2, N)`), no dtype constraint, no validation at any point. Subclasses directly assign `self.zero_loc = <anything>`.
- **`shape` has no validation**: Accepts any tuple, including `(0, 0)`, `(-1, -1)`, or `("a", "b")`.
- **`mtype` is a bare string**: Should be an enum or Literal. Currently set to "base", "list", "file", "threshold", "parameter", "array" across subclasses.
- **`qrings` is an untyped list**: `self.qrings: list = []` — no element type constraint.
- **No ABC enforcement**: `evaluate()` has a default `pass` implementation instead of `@abstractmethod`. Subclasses can silently forget to override.
- **`get_mask()` reconstructs the mask from `zero_loc` every call**: No caching, no shape validation that `zero_loc` indices are within `self.shape`.
- **`combine_mask()` mutates its argument**: `mask[tuple(self.zero_loc)] = 0` modifies the passed array in place, which is a side effect not obvious from the method signature.

**Recommendation:**
1. Make `MaskBase` an ABC with `@abstractmethod` on `evaluate()`.
2. Validate `zero_loc` shape as `(2, N)` and dtype as integer in a setter or property.
3. Validate `shape` as positive integers.
4. Use `MaskType` enum instead of `mtype` string.

#### MaskList (2/1/3/1)

Inherits all MaskBase weaknesses. Additional issues:
- `append_zero_pt()` uses `np.append()` which copies the entire array on each call — O(N) per append. For large masks with many points, this is O(N^2) total.
- `xylist` attribute is declared but never used.
- `evaluate()` directly assigns `self.zero_loc = zero_loc` with no validation.

#### MaskThreshold (3/2/3/2)

Slightly better than MaskBase:
- Uses backend abstraction (`get_backend()`, `ensure_numpy()`) correctly
- Handles `saxs_lin is None` gracefully
- But still no validation on `low`/`high` thresholds (e.g., `low > high` would create an empty mask silently)

#### MaskAssemble (3/2/4/2)

**What it does well:**
- Bounded history with `max_history` parameter prevents unbounded memory growth
- `_get_mask_ref()` returns read-only view for internal use (performance optimization)
- `get_mask()` returns a copy for external callers (safety)
- Change detection short-circuit in `apply()` avoids unnecessary history entries
- `redo_undo()` navigation is correct with `mask_ptr_min` anchor

**What's weak:**
- **`workers` dict typed as `dict[str, Any]`**: The worker names are hardcoded string keys with no enum or constant backing. A typo in `self.workers["mask_drow"]` silently returns `KeyError` at runtime.
- **No shape enforcement on history**: `mask_record` is `list[np.ndarray]` but no invariant ensures all entries have the same `self.shape`. If a worker returns a mask with a different shape, the logical AND operation may broadcast silently or raise a confusing error.
- **`evaluate()` dispatches on string keys**: `if target == "mask_threshold": ...` — this should be polymorphic dispatch.

Cross-reference: "MaskAssemble history stack has no type enforcement on mask array shape/dtype" noted in contract_audit.md context.

---

### 2.4 Backend Types (`xpcsviewer/backends/`)

#### BackendProtocol (5/3/5/3)

**What it does well:**
- `@runtime_checkable Protocol` enables structural typing — both `JAXBackend` and `NumPyBackend` satisfy it without explicit inheritance
- Comprehensive API surface: 60+ methods covering array creation, math, statistics, type conversion, JIT, grad, vmap
- `to_numpy()` and `from_numpy()` are included in the protocol (corrects C-013 from contract audit — this was already fixed)

**What's weak:**
- **`ArrayType` is a TypeVar, not a concrete type**: `ArrayType = TypeVar("ArrayType", bound=...)` — callers receive `ArrayType` return types but cannot narrow to `np.ndarray` or `jnp.ndarray` without concrete backend knowledge. This is inherent to the protocol pattern but limits downstream type safety.
- **No error contract**: Methods don't declare what exceptions they raise. `grad()` on NumPyBackend presumably raises `NotImplementedError`, but the protocol doesn't express this.
- **Return types are all `ArrayType`**: No distinction between scalar returns (from `sum()`, `mean()`) and array returns. `backend.sum(x)` returns `ArrayType` but may be a scalar.

#### JAXBackend (4/3/5/3)

**What it does well:**
- Clean implementation of `BackendProtocol` with direct delegation to `jnp.*`
- `_ensure_jax()` guard in `__init__` ensures JAX is available before use
- Lazy JAX import pattern (`_jax`, `_jnp` module-level) avoids import-time side effects

**What's weak:**
- **Stateless singleton**: No configuration state, no device tracking. The backend doesn't know which device it's using. Device selection is handled externally via JAX environment variables.
- **No cache or compilation state management**: JIT-compiled functions are managed externally (in `qmap.py` via `_JIT_CACHE`). The backend itself doesn't provide a JIT cache interface.

---

### 2.5 Threading Types (`xpcsviewer/threading/`)

#### WorkerState (5/5/3/5) / TaskPriority (5/5/4/5) / TaskType (5/5/4/5)

All three are clean enums:
- Closed set of values
- `WorkerState` has 7 states covering the full lifecycle
- `TaskPriority` uses integer values for priority queue ordering
- `TaskType` categorizes tasks for pool dispatch

Note: `WorkerState` has both `ERROR` and `FAILED` — the distinction is unclear from the enum alone. `WorkerState` is defined but its usage is limited (not referenced from `BaseAsyncWorker.run()` which uses ad-hoc state tracking).

#### WorkerResult (4/3/3/3)

**What it does well:**
- Validates `execution_time >= 0` and `memory_usage >= 0`
- Simple, flat structure

**What's weak:**
- `data: Any` — completely untyped. Callers must know the concrete worker type to interpret the result.
- `error: str` — always required, even for successful results. Should be `str | None` with `error` required only when `success=False`.
- No invariant linking `success` to `error` presence: `WorkerResult(success=True, data=None, error="oops", ...)` is valid.

**Recommendation:** Add `__post_init__` validation: if `success is True`, `error` should be empty; if `success is False`, `error` should be non-empty.

#### WorkerStats (4/3/3/3)

Clean monitoring dataclass with thorough non-negative validation. Same pattern as `WorkerResult`.

**What's weak:**
- No invariant: `completed_jobs + failed_jobs <= total_jobs` is not enforced.
- `average_execution_time` is stored as a raw field rather than computed from totals — can become stale.

#### WorkerSignals (3/2/4/2)

**What it does well:**
- Clear signal declarations with type comments explaining each signal's parameters
- Comprehensive coverage: started, finished, error, progress, status, cancelled, partial_result, resource_usage, state_changed, retry_attempt

**What's weak:**
- **Signal parameter types are stringly-typed**: `Signal(str, int, int, str, float)` — the meaning of each parameter is only documented in comments. PySide6 signals don't support named parameters.
- **`finished = Signal(object)`** — the result type is `object`. Callers must cast.
- **`resource_usage` and `state_changed` signals are declared but never emitted** by `BaseAsyncWorker.run()`. Dead signals.

#### BaseAsyncWorker (3/2/5/2)

**What it does well:**
- Clean lifecycle: `run()` catches all exceptions, emits appropriate signals
- Rate-limited progress emission (10/sec) with ETA caching
- Performance stats collection
- Thread-safe cancellation via `threading.Event`

**What's weak:**
- **Many exposed internal fields**: `_last_progress_time`, `_cached_eta`, `_progress_emission_interval`, etc. are public despite underscore convention. Any subclass can mutate them.
- **`do_work()` returns `Any`**: No type parameterization. A `PlotWorker` and a `DataLoadWorker` both return `Any`.
- **No state machine**: The `WorkerState` enum exists but `BaseAsyncWorker` doesn't use it. State transitions (idle -> running -> completed/error/cancelled) are implicit.
- **Signal connection pattern in `run()`**: `self.signals.finished.emit(result)` passes `result` as `object`, losing type information.

**Recommendation:** Parameterize `BaseAsyncWorker` with a generic result type: `class BaseAsyncWorker(QRunnable, Generic[T])` with `do_work() -> T` and `signals.finished = Signal(T)`. (PySide6 limitation: signals can't be generic, but the return type can be.)

#### WorkerManager (3/2/4/2)

**What it does well:**
- Thread-safe with `_workers_lock` for concurrent access
- Clean lifecycle: submit, cancel, get result/error, shutdown
- Signal forwarding from workers to manager-level signals

**What's weak:**
- **`worker_results: dict[str, Any]`** and **`worker_errors: dict[str, tuple[str, str]]`**: Results are stringly-keyed and untyped. No connection between a worker's ID and its result type.
- **Results accumulate forever**: `worker_results` and `worker_errors` dicts grow without bound. No TTL, no eviction, no size limit.
- **Signal connections in `submit_worker()`** use lambdas that capture `worker_id` — these lambda closures keep the worker reference alive, preventing garbage collection until the manager is destroyed.

#### UnifiedTask (3/3/4/3)

**What it does well:**
- Validates `task_id` non-empty, `timeout_seconds > 0`, `memory_limit_mb > 0`
- Priority queue ordering via `__lt__`
- Clean computed properties: `execution_time`, `wait_time`

**What's weak:**
- **`function: Callable`** — completely untyped callable. No signature constraint.
- **`result: Any`** and **`error: Exception | None`** — mutable state on what is otherwise a value object.
- **`args: tuple` and `kwargs: dict`** — untyped, cannot be validated against `function`'s signature.

#### UnifiedThreadingManager (3/2/4/2)

**What it does well:**
- Memory-aware task scheduling via `_can_accept_task()`
- Lazy thread pool creation with double-checked locking
- Queue-based signal emission for thread safety (BUG-007 fixes)
- System monitoring thread with adaptive pool sizing

**What's weak:**
- **`_adjust_pool_size()` is a no-op**: `ThreadPoolExecutor` doesn't support runtime resizing. The method logs what it "would do" but does nothing. This is dead code masquerading as a feature.
- **`_stats` dict is mutable with no schema**: Plain `dict[str, int | float]` — a dataclass or TypedDict would provide type safety.
- **`_completed_tasks` grows to 50 entries then trims**: The trimming logic sorts by `completed_at` on every completion, which is O(N log N). Should use a bounded deque.
- **Thread pool configs are hardcoded dicts**: `_pool_configs` uses `dict[str, Any]` instead of a typed config dataclass.

---

### 2.6 XpcsFile (`xpcsviewer/xpcs_file.py`)

#### XpcsFile (1/1/5/1)

**What it does well:**
- Essential type — the primary data container for the entire application
- Comprehensive API: 40+ methods covering SAXS, G2, twotime, fitting, ROI, memory management
- Protected attribute set (`_PROTECTED`) prevents HDF5 payload from overwriting instance attrs (BUG-015)
- Context manager support (`__enter__`/`__exit__`/`close`)
- UUID-based instance ID for cleanup tracking

**What's critically weak:**
- **God object**: 50+ attributes, 40+ methods, 800+ lines. Violates single-responsibility principle.
- **`__getattr__` dynamic dispatch (C-007, C-008)**: Routes attribute access through a chain of string comparisons (`_QMAP_KEYS`, `"saxs_2d"`, `"Int_t_fft"`, `_G2_PARTIAL_KEYS`, `__dict__`). This makes the type's interface invisible to static analysis.
- **No type annotation on constructor parameters**: `__init__(self, fname, fields=None, label_style=None, qmap_manager=None)` — all parameters untyped.
- **Payload dictionary unpacked into `__dict__`**: `self.__dict__[k] = v` — attributes are defined dynamically from HDF5 file contents. The set of attributes varies per file.
- **`self.qmap` can be None**: If `get_qmap()` returns None (e.g., file has no Q-map), `__getattr__` delegates to `getattr(self.qmap, key)` which raises `AttributeError("'NoneType'...")` (C-007).
- **`_G2_PARTIAL_KEYS` lazy load stores None on failure** (C-008): Downstream callers receive None without type safety.
- **`fit_g2()` uses `assert` for bounds validation** (C-009): Disabled with `-O` flag.
- **No schema for worker result dicts**: `apply_saxs_2d_result()`, `apply_twotime_result()` use hardcoded dict keys with no TypedDict or schema.

**Recommendation:**
1. Extract sub-interfaces: `XpcsSaxsData`, `XpcsG2Data`, `XpcsTwotimeData`, `XpcsFitting` — each with typed fields and methods.
2. Replace `__getattr__` dispatch with explicit properties or a typed `Namespace` object.
3. Replace `assert` with `if/raise ValueError`.
4. Define TypedDicts for worker result contracts.
5. Add type annotations to `__init__` parameters.

Cross-reference: C-006, C-007, C-008, C-009 in `contract_audit.md`. P1-003 in `numerical_audit.md`.

---

## 3. Critical Types (Score <= 2 in any category)

### Priority 1 — Types with Enforcement = 1

| Type | Score | Key Issue | Fix |
|------|-------|-----------|-----|
| **XpcsFile** | 1/1/5/1 | God object with dynamic attrs, no validation | Extract typed sub-interfaces; add `__init__` annotations |
| **MaskBase** | 2/1/3/1 | `zero_loc` untyped, no shape validation | Add `zero_loc` property with shape/dtype validation; make ABC |
| **MaskList** | 2/1/3/1 | Inherits MaskBase weaknesses | Fix MaskBase; add `zero_loc` validation in `evaluate()` |

### Priority 2 — Types with Invariant Expression <= 2

| Type | Score | Key Issue | Fix |
|------|-------|-----------|-----|
| **NLSQResult** | 2/2/4/2 | 540 lines of boilerplate delegation; `health_score`/`condition_number` unguarded attribute chains | Extract delegation into descriptor; add `hasattr` guards |
| **MaskAssemble** | 3/2/4/2 | Workers dict stringly-typed; no shape enforcement on history | Use `MaskWorkerType` enum for keys; validate mask shapes |
| **WorkerSignals** | 3/2/4/2 | Signal params stringly-typed; dead signals | Remove unused signals; add comment-level type docs |
| **WorkerManager** | 3/2/4/2 | Untyped result dict; no result eviction | Add TypedDict for results; add size limit |
| **UnifiedThreadingManager** | 3/2/4/2 | Dead `_adjust_pool_size`; untyped `_stats` dict | Remove dead code; use `@dataclass` for stats |
| **BaseAsyncWorker** | 3/2/5/2 | No state machine; `do_work()` returns `Any` | Add WorkerState tracking; parameterize return type |

---

## 4. Cross-References to Previous Audit Findings

### From `contract_audit.md`

| Finding | Type Affected | Type Design Implication |
|---------|--------------|------------------------|
| C-003: `health_score` unguarded `diagnostics.health_score` | `NLSQResult` | Invariant score = 2 (delegation chain not type-safe) |
| C-004: `condition_number` unguarded chain | `NLSQResult` | Same as above |
| C-005: `read_qmap()` returns dict when `validate=False` | `QMapSchema` (consumer) | Schema enforcement is optional — HDF5Facade return type is a lie |
| C-007: `__getattr__` delegates to None qmap | `XpcsFile` | Encapsulation score = 1 (no typed interface) |
| C-008: `_G2_PARTIAL_KEYS` stores None silently | `XpcsFile` | Enforcement score = 1 (no type safety on lazy attrs) |
| C-009: `assert` for bounds validation | `XpcsFile` | Enforcement score = 1 (disabled with `-O`) |
| C-010: Empty FitDiagnostics.converged = True | `FitDiagnostics` | Invariant score = 3 (vacuous truth on empty state) |
| C-015: G2Data NaN check too strict | `G2Data` | Invariant score = 4 (correct but overly strict) |
| C-028: FitResult.to_dict() never called | `FitResult` | Usefulness score = 4 (serialization exists but unused) |

### From `numerical_audit.md`

| Finding | Type Affected | Type Design Implication |
|---------|--------------|------------------------|
| P1-001: `converged` always False | `NLSQResult` | `converged` field is set from outside, not validated internally |
| P1-003: `model_fn(x, *popt)` JIT retrace | `NLSQResult` | `params` dict stores NumPy scalars, not native floats |
| P2-002: `lru_cache` returns mutable dicts | `QMapSchema` (consumer) | The cache returns raw dicts, not frozen schemas |

---

## 5. Summary Recommendations (Ranked by Impact)

### High Impact

1. **Refactor XpcsFile into typed sub-interfaces** — Extract `XpcsSaxsData`, `XpcsG2Data`, `XpcsTwotimeData` interfaces with typed fields. Replace `__getattr__` with explicit properties. This addresses C-007, C-008, C-009 and makes the type inspectable by static analysis.

2. **Fix FitDiagnostics vacuous truth** — Add `if not self.r_hat and not self.ess_bulk: return False` to prevent empty diagnostics from reporting convergence. This addresses C-010.

3. **Add hasattr guards in NLSQResult delegation** — Guard `health_score` and `condition_number` against missing attributes on the NLSQ native result. This addresses C-003 and C-004.

4. **Make MaskBase an ABC with validated zero_loc** — Add `@abstractmethod` on `evaluate()`, validate `zero_loc` shape as `(2, N)` in a property setter. This prevents all mask subclass invariant violations.

### Medium Impact

5. **Extract NLSQResult delegation boilerplate** — Replace 11 repeated getter/setter pairs with a descriptor class or `__getattr__` pattern. Reduces 540 lines to ~100.

6. **Type MaskAssemble workers dict** — Replace string keys with a `MaskWorkerKey` enum. Add shape validation on mask history entries.

7. **Implement FitResult persistence** — Add `from_dict()` and connect to session save/restore. Addresses C-028.

8. **Add TypedDict for worker result contracts** — Define `SaxsResult`, `G2Result`, `TwotimeResult` TypedDicts for the dict-based communication between workers and the GUI layer.

### Low Impact

9. **Remove dead WorkerSignals** — `resource_usage` and `state_changed` are never emitted.
10. **Remove dead `_adjust_pool_size`** in `UnifiedThreadingManager`.
11. **Use WorkerState enum in BaseAsyncWorker** — The enum exists but isn't used for state tracking.

---

*Phase 4 type design audit complete. 25 types analyzed, 6 critical types flagged (score <= 2), 11 recommendations prioritized.*
