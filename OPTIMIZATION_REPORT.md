# OPTIMIZATION REPORT — XPCSViewer Performance Swarm

Date: 2026-03-03

## Executive Summary

A 5-agent swarm performed end-to-end performance analysis and optimization of the xpcsviewer
codebase (137 Python files, ~60,000 lines). The effort uncovered one critical correctness bug
in the streaming SAXS log path, delivered one high-impact GPU kernel replacing the primary
SAXS render path, reduced PyQtGraph render overhead in G2 plot functions, added a JAX-gated
path to a twotime statistics function, and repaired four defects in an existing parallel file
loading path.

| Metric | Value |
|---|---|
| Files analyzed | 137 |
| Hotspots profiled | 5 |
| Bugs fixed | 1 (BUG-001 — critical correctness, 4-order-of-magnitude error) |
| Optimizations implemented | 4 (OPT-001, OPT-002, OPT-003, OPT-004) |
| Optimizations closed/no-gain | 1 (OPT-005) |
| Primary speedup | 2.2–4.51× SAXS log render (OPT-004, GPU) |
| Unit tests before → after | 812/813 → 812/813 (no regression) |

---

## Environment

| Property | Value |
|---|---|
| Python | 3.13.7 |
| JAX backend | GPU — `CudaDevice(id=0)` |
| NumPy | 2.3.5 |
| PySide6 | 6.10.2 |
| Platform | Linux 6.8.0 x86_64 |
| Package manager | uv |

---

## Changes Made

### BUG-001 — SAXSLogProcessor global min fix [COMPLETE]

**File:** `xpcsviewer/utils/streaming_processor.py`
**Severity:** Critical — 4-orders-of-magnitude log-scale error on beamstop pixels

**Root cause:** `SAXSLogProcessor.process_chunk` used the *per-chunk local minimum* as the
non-positive pixel replacement floor. For XPCS data with beamstops, non-positive pixels are
concentrated in chunk 0 (beam centre), while the minimum positive intensity is in peripheral
chunks. The floor applied to beamstop pixels was therefore `log10(local_high_min) ≈ +1`
instead of the correct `log10(global_min) ≈ −3`.

**Fix:** Overrode `process_array_streaming` in `SAXSLogProcessor` with a two-pass scan:
- **Pass 1** — iterate all chunks to compute `global_min` across the full array
- **Pass 2** — delegate to `super().process_array_streaming` with `self._global_min` set
  consistently; `process_chunk` now uses this single floor value for every chunk

**Correctness:** `_compute_saxs_log_standard` (NumPy) and the new JAX kernel were already
correct — they operate on the full array globally. Only the streaming path (>10 M pixels)
was affected.

```
Before: max log-scale error = 4.0 (beamstop pixels: +1 instead of −3)
After:  max log-scale error = 0.00 (5/5 tests, including beamstop pattern)
```

---

### OPT-004 — SAXS log JAX GPU kernel [COMPLETE]

**Files:** `xpcsviewer/xpcs_file.py`
**Implemented by:** jax agent + team-lead

Added a module-level `@jax.jit` lazy singleton kernel and wired both production call sites to
use the JAX path with NumPy fallback.

- `_get_jax_saxs_log()` (lines 89–108): compiled once per session on first call. Computes
  `jnp.maximum(data, min_pos)` → `jnp.log10(safe).astype(jnp.float32)`. `jnp.isinf` guard
  handles all-zero arrays.
- `_compute_saxs_log_jax()` (lines 843–861): instance wrapper; falls back to
  `_compute_saxs_log_standard` on any JAX failure.
- Both call sites (lines 579, 795) now use `_compute_saxs_log_jax`.

**Benchmark (5 warmup + 50 timed runs, GPU):**

| Shape | NumPy baseline | JAX GPU | Speedup |
|---|---|---|---|
| 512×512 | 1.6 ms | 0.7 ms | **2.2×** |
| 1024×1024 | 7.2 ms | 3.3 ms | **2.2×** |
| 2048×2048 | 35–56 ms | 9–12.5 ms | **2.2–4.51×** |

- max_diff vs NumPy: **4.77e-07** (well within float32 tolerance)
- JIT compile cost: 220 ms one-time, amortized after first call

---

### OPT-001 — g2mod Qt overhead reduction [COMPLETE]

**File:** `xpcsviewer/module/g2mod.py`
**Lines modified:** ~211-271, ~403-531, ~648-740 (three plot functions)
**Owner:** python agent

The inner `for n in range(num_qval)` loops cannot be broadcast away — each iteration makes
per-Q PyQtGraph API calls that are inherently serial. Optimization focused on eliminating
redundant Python work:

- `ax.setLabel("bottom"/"left")` hoisted from the m×n inner loop to the `num_figs` setup
  loop — reduces Qt calls from `num_data × num_qval` to `num_figs` per render
- `color` and `symbol` modulo lookups hoisted from inner n loop to outer m loop
- `color_len = len(colors)` and `symbol_len = len(symbols)` precomputed once
- `fit_summary = None` initialization added before conditional block (correctness fix —
  prevents potential NameError if `show_fit=False`)

**Measured reduction (8 files × 32 Q):** setLabel calls: 512 → 64 (87.5% reduction);
color/symbol lookups: 256 → 16 each. Actual wall-clock improvement is Qt-overhead-bound
rather than NumPy-compute-bound; smaller than the 3.3× profiled data-prep target.
72/72 unit tests pass.

---

### OPT-002 — twotime C2 JAX gating [COMPLETE — dead code, future use]

**File:** `xpcsviewer/module/twotime_utils.py`
**Lines modified:** ~639-716 (`compute_c2_statistics_vectorized`, 37 → 75 lines)
**Owner:** python agent

JAX-accelerated path added to `compute_c2_statistics_vectorized`, gated on matrix dimension
`n >= 1024`. Uses `jnp` for `mean`/`std`/`min`/`max`/`diagonal`/`sum`; `np.median` retained
(no JAX equivalent). `try/except` fallback to NumPy on any JAX error.

**Important:** `compute_c2_statistics_vectorized` has **no production callers** — it is dead
code in the `xpcsviewer/` package (only referenced from `tests/benchmarks/`). The JAX gate is
correctly implemented and will benefit benchmark tests; **zero wall-clock impact on user
workflows** until the function is wired to a production call site.

Profiled speedup at 2048×2048: 8.8× (JAX vs NumPy). NumPy path unchanged for n < 1024
(JAX was 0.7× slower at 512×512 due to transfer overhead). 72/72 unit tests pass.

---

### OPT-003 — average_toolbox parallel path defect fixes [COMPLETE]

**File:** `xpcsviewer/module/average_toolbox.py`
**Owner:** systems agent

**Finding:** `_process_files_parallel` + `_process_batch` already exist and are dispatched
for N≥8 files. h5py holds the GIL during reads — threading cannot overlap HDF5 I/O (measured
2 threads = 0.43× sequential). Benefit comes from CPU-side numpy accumulation overlap (~1.37×
at N=20 warm files).

**Four defects fixed in the existing parallel path:**

| # | Defect | Type | Fix |
|---|---|---|---|
| 1 | No `is_killed` check in `_process_batch` | Correctness | Added per-file check at top of batch loop |
| 2 | `baseline[]` written in completion order (not file order) | Correctness | Use `self.baseline[m]`; set `self.ptr = tot_num` after executor |
| 3 | No `xf.clear_cache()` in batch worker | Memory | Mirror sequential path cleanup after each file |
| 4 | `n_workers = cpu_count()` — GIL limits beyond 4 | Performance | Capped at `min(len(batches), 4)` |

---

### OPT-005 — twotime `np.array` stacking [CLOSED — no gain]

**File:** `xpcsviewer/module/twotime_utils.py`

Profiled: `np.stack` vs `np.array([list...])` — 1.02× difference, not measurable. Closed
without action.

---

## Performance Summary Table

| ID | Description | File | Speedup | Status |
|---|---|---|---|---|
| BUG-001 | SAXSLogProcessor global min | `utils/streaming_processor.py` | Correctness fix | COMPLETE |
| OPT-004 | SAXS log JAX GPU kernel | `xpcs_file.py` | 2.2–4.51× | COMPLETE |
| OPT-001 | g2mod Qt overhead reduction | `module/g2mod.py` | 87.5% fewer setLabel Qt calls | COMPLETE |
| OPT-002 | twotime C2 JAX gating | `module/twotime_utils.py` | 8.8× at 2048×2048 (dead code — future) | COMPLETE |
| OPT-003 | average_toolbox parallel defects (4 fixes) | `module/average_toolbox.py` | Correctness + memory + ~1.37× | COMPLETE |
| OPT-005 | twotime np.array stacking | `module/twotime_utils.py` | 1.02× — none | CLOSED |

---

## Correctness Verification

### BUG-001

| Test case | Shape | max_err | Result |
|---|---|---|---|
| Beamstop pattern (fine chunks) | 200×200 | 0.00e+00 | PASS |
| Uniform positive array | 512×512 | 0.00e+00 | PASS |
| All-zero edge case | 100×100 | 0.00e+00 | PASS |
| `process_saxs_log_streaming` wrapper | 200×200 | 0.00e+00 | PASS |
| Benchmark array (2048×2048) | 2048×2048 | 0.00e+00 | PASS |

Streaming two-pass overhead: median 96 ms, p95 108 ms at 2048×2048 with chunk_size_mb=50.
Acceptable — this code path activates only for detectors >10 M pixels.

### OPT-004

| Test case | Shape | max_diff vs NumPy | Result |
|---|---|---|---|
| Beamstop pattern | 512×512 | 1.19e-07 | PASS |
| All-positive array | 1024×1024 | < float32 eps | PASS |
| Benchmark (5 warmup + 50 runs) | 2048×2048 | **4.77e-07** | PASS |

Tolerance used: rtol=1e-4, atol=1e-7 (float32 GPU/CPU).

### Unit Test Suite

| Metric | Before Phase 3 | After Phase 3 |
|---|---|---|
| Tests run | 813 | 813 |
| Passed | 812 | 812 |
| Failed | 1 | 1 |
| New failures | — | **0** |

The single failure (`TestXpcsFileAttributeCollision::test_xpcsfile_load_data_result_checked_for_collision`)
is a pre-existing test ordering / state-isolation issue: it passes when run in isolation and
also fails on the baseline commit with all Phase 3 changes stashed. **No regression introduced.**

---

## Known Issues

- **Pre-existing test isolation failure:** `TestXpcsFileAttributeCollision::test_xpcsfile_load_data_result_checked_for_collision` fails in full suite but passes in isolation. Confirmed pre-existing by git-stash test on baseline. Not a Phase 3 regression. Requires separate investigation of test ordering / fixture teardown.

---

## Files Modified

| File | Change | Phase |
|---|---|---|
| `xpcsviewer/utils/streaming_processor.py` | BUG-001: two-pass global-min scan in `SAXSLogProcessor.process_array_streaming` | 3 |
| `xpcsviewer/xpcs_file.py` | OPT-004: `_get_jax_saxs_log` module-level JIT kernel + `_compute_saxs_log_jax` instance method; both call sites wired | 3 |
| `xpcsviewer/module/g2mod.py` | OPT-001: setLabel/color/symbol hoisted from inner loop; fit_summary NameError fix | 3 |
| `xpcsviewer/module/twotime_utils.py` | OPT-002: JAX path gated on n≥1024 in `compute_c2_statistics_vectorized` | 3 |
| `xpcsviewer/module/average_toolbox.py` | OPT-003: 4 defect fixes in `_process_files_parallel` / `_process_batch` | 3 |
| `OPTIMIZATION_LEDGER.md` | Full profiling ledger with baselines, Phase 4 table, agent handoff record | 1–4 |

---

*Generated by Performance Optimization Swarm — xpcsviewer — 2026-03-03*
