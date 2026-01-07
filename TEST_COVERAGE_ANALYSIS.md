# Legacy Component Test Coverage Analysis - XPCS Viewer

**Analysis Date:** 2026-01-06
**Branch:** 001-jax-migration
**Repository:** xpcsviewer

---

## Executive Summary

Analysis of legacy/god-class modules in XPCS Viewer reveals **critical coverage gaps** across core functionality. The codebase has identified 4 major god-object classes with high public method counts (28-80 public methods each) but insufficient test coverage for critical paths.

### Coverage Statistics

| Module | Total Public Methods | Test Methods | Estimated Coverage | Priority |
|--------|----------------------|--------------|-------------------|----------|
| xpcsviewer/xpcs_file.py | 39 | 39* | ~5% actual** | P1 |
| xpcsviewer/viewer_kernel.py | 28 | 50 | ~5% actual** | P1 |
| xpcsviewer/module/g2mod.py | 11 | 82 | ~35% | P1 |
| xpcsviewer/module/twotime.py | 4 | 30 | ~30% | P2 |
| xpcsviewer/fileIO/hdf_reader.py | 25+ | 85 | ~20% | P1 |
| xpcsviewer/fileIO/qmap_utils.py | 20+ | 50 | ~40% | P2 |
| xpcsviewer/module/saxs1d.py | 10 | 24 | ~40% | P2 |

*Tests exist but mostly test indirect dependencies, not the main methods
**Actual coverage based on direct method testing (methods called by test methods)

---

## Module-by-Module Analysis

### 1. xpcsviewer/xpcs_file.py (GOD OBJECT)

**Status:** CRITICAL - Core module with 39 public methods, <5% actual coverage

#### Overview
- **File Path:** `/Users/b80985/Projects/xpcsviewer/xpcsviewer/xpcs_file.py`
- **Class:** `XpcsFile` (80 total items including properties and setters)
- **Test File:** `/Users/b80985/Projects/xpcsviewer/tests/unit/core/test_xpcs_file.py`
- **Test Count:** 39 test methods (but mostly indirect)

#### Public Methods (39 total)

**Tested Methods (2 of 39 = 5%):**
- `get_hdf_info(fstr=None)` - Basic test coverage
- `update_label(label_style)` - Basic test coverage

**Untested Critical Methods (37 of 39):**

*Data Access Methods:*
- `get_g2_data(qrange=None, trange=None)` - Core analysis function, **UNTESTED**
- `get_g2_stability_data(qrange=None, trange=None)` - Stability analysis, **UNTESTED**
- `get_twotime_c2(selection=0, correct_diag=True, max_size=32678)` - TwoTime data, **UNTESTED**
- `get_twotime_stream()` - Memory-efficient streaming, **UNTESTED**
- `get_cropped_qmap(target="dqmap", enabled=True)` - Q-map processing, **UNTESTED**
- `get_offseted_g2(normalization=False)` - G2 normalization, **UNTESTED**

*Analysis Methods:*
- `fit_g2(...)` - Core fitting functionality, **UNTESTED**
- `fit_g2_robust(...)` - Robust fitting, **UNTESTED**
- `fit_g2_high_performance(...)` - Performance variant, **UNTESTED**
- `fit_tauq(q_range, bounds, fit_flag, force_refit=False)` - Q-dependent fitting, **UNTESTED**
- `correct_g2_err(g2_err=None, threshold=1e-6)` - Error correction, **UNTESTED**

*ROI and Export Methods:*
- `get_roi_data(roi_parameter, phi_num=180)` - ROI extraction, **UNTESTED**
- `get_multiple_roi_data_parallel(roi_list, phi_num=180, max_workers=None)` - Parallel ROI, **UNTESTED**
- `export_saxs1d(roi_list, folder)` - SAXS 1D export, **UNTESTED**
- `get_pg_tree()` - GUI tree structure, **UNTESTED**

*Cache Management:*
- `clear_cache(cache_type: str = "all")` - Memory management, **UNTESTED**
- `get_cache_stats()` - Performance monitoring, **UNTESTED**

#### Critical Paths Lacking Tests

1. **G2 Analysis Pipeline** (High Risk)
   - Load → Normalize → Fit → Export
   - Methods: `get_g2_data()` → `fit_g2()` → `export_*`
   - Impact: Core XPCS analysis feature

2. **TwoTime Analysis** (High Risk)
   - Load → Correct → Stream → Export
   - Methods: `get_twotime_c2()` → `get_twotime_stream()` → export
   - Impact: TwoTime correlation analysis

3. **ROI Extraction** (High Risk)
   - Single/Multiple ROI extraction with phi binning
   - Methods: `get_roi_data()`, `get_multiple_roi_data_parallel()`
   - Impact: Custom ROI analysis

4. **Memory Management** (Medium Risk)
   - Cache operations during large file processing
   - Methods: `clear_cache()`, `get_cache_stats()`
   - Impact: Stability under memory pressure

#### Estimated Coverage

```
Breakdown:
- Memory Monitor tests:     ✓ 4/4 methods tested (100%)
- Initialization tests:     ✓ 3/3 variations tested (100%)
- String representation:    ✓ 2/2 methods tested (100%)
- Data access methods:      ✗ 0/12 methods tested (0%)
- Analysis/fitting:         ✗ 0/8 methods tested (0%)
- ROI/Export:              ✗ 0/6 methods tested (0%)
- Cache management:        ✗ 0/3 methods tested (0%)

ACTUAL COVERAGE: ~5% of public methods
```

#### Recommended Priority Test Additions (P1)

**High Priority (Blocks Release):**
1. `test_get_g2_data_basic()` - Basic G2 loading
2. `test_fit_g2_single_exponential()` - Single-exp fit
3. `test_fit_g2_double_exponential()` - Double-exp fit
4. `test_get_g2_stability_data()` - Stability analysis
5. `test_get_twotime_c2()` - TwoTime loading
6. `test_get_roi_data_single()` - Single ROI extraction
7. `test_get_multiple_roi_data_parallel()` - Parallel ROI
8. `test_export_saxs1d()` - SAXS 1D export

**Medium Priority:**
9. `test_fit_tauq()` - Q-dependent tau fitting
10. `test_clear_cache()` - Cache operations
11. `test_get_cache_stats()` - Cache stats
12. `test_get_offseted_g2()` - G2 normalization

---

### 2. xpcsviewer/viewer_kernel.py (GOD OBJECT)

**Status:** CRITICAL - Analysis orchestrator with 28 public methods, <5% actual coverage

#### Overview
- **File Path:** `/Users/b80985/Projects/xpcsviewer/xpcsviewer/viewer_kernel.py`
- **Class:** `ViewerKernel` (analysis orchestration)
- **Test File:** `/Users/b80985/Projects/xpcsviewer/tests/unit/core/test_viewer_kernel.py`
- **Test Count:** 50 test methods (but only 1 method actually tested)

#### Public Methods (28 total)

**Tested Methods (1 of 28 = 4%):**
- `reset_meta()` - State reset, **Tested**

**Untested Critical Methods (27 of 28):**

*Plotting Methods (Core Functionality):*
- `plot_g2(handler, q_range, t_range, y_range, rows=None, **kwargs)` - G2 plotting, **UNTESTED**
- `plot_g2_stability(mp_hdl, rows=None, **kwargs)` - G2 stability plot, **UNTESTED**
- `plot_g2map(hdl, rows=None, target=None)` - G2 map plotting, **UNTESTED**
- `plot_saxs_1d(pg_hdl, mp_hdl, **kwargs)` - SAXS 1D plotting, **UNTESTED**
- `plot_saxs_2d(hdl, rows=None, **kwargs)` - SAXS 2D plotting, **UNTESTED**
- `plot_twotime(hdl, rows=None, **kwargs)` - TwoTime plotting, **UNTESTED**
- `plot_intt(pg_hdl, rows=None, **kwargs)` - Intensity-Time plotting, **UNTESTED**
- `plot_stability(mp_hdl, rows=None, **kwargs)` - Stability plotting, **UNTESTED**
- `plot_qmap(hdl, rows=None, target=None)` - Q-map plotting, **UNTESTED**

*ROI/Geometry Methods:*
- `add_roi(hdl, **kwargs)` - ROI addition, **UNTESTED**
- `plot_tauq_pre(hdl=None, rows=None)` - Tau-Q preprocessing, **UNTESTED**

*Export Methods:*
- `export_g2(folder, rows=None)` - G2 export, **UNTESTED**
- `export_saxs_1d(pg_hdl, folder)` - SAXS 1D export, **UNTESTED**
- `export_diffusion(folder, rows=None)` - Diffusion data export, **UNTESTED**

*Data Retrieval Methods:*
- `get_info_at_mouse(rows, x, y)` - Mouse hover information, **UNTESTED**
- `get_pg_tree()` - PyQtGraph parameter tree, **UNTESTED**
- `get_fitting_tree()` - Fitting parameter tree, **UNTESTED**
- `get_memory_stats()` - Memory statistics, **UNTESTED**

*Job Management:*
- `submit_job(*args, **kwargs)` - Job submission, **UNTESTED**
- `remove_job(index)` - Job removal, **UNTESTED**
- `update_avg_info(jid)` - Average job info, **UNTESTED**
- `update_avg_values(data)` - Average job values, **UNTESTED**

*Utility Methods:*
- `switch_saxs1d_line(mp_hdl, lb_type)` - SAXS 1D line switching, **UNTESTED**
- `get_module()` - Module getter, **UNTESTED**

#### Critical Paths Lacking Tests

1. **G2 Analysis Plotting** (Critical)
   - Plot generation for G2 data
   - Methods: `plot_g2()`, `plot_g2_stability()`, `plot_g2map()`
   - Impact: Core visualization feature

2. **SAXS Analysis Plotting** (Critical)
   - SAXS 1D/2D plotting and ROI operations
   - Methods: `plot_saxs_1d()`, `plot_saxs_2d()`, `add_roi()`
   - Impact: SAXS analysis workflow

3. **Data Export** (High Risk)
   - Export to various formats
   - Methods: `export_g2()`, `export_saxs_1d()`, `export_diffusion()`
   - Impact: Data pipeline and reproducibility

4. **Job Management** (Medium Risk)
   - Averaging and batch operations
   - Methods: `submit_job()`, `update_avg_info()`, `update_avg_values()`
   - Impact: Long-running operations

#### Estimated Coverage

```
Breakdown:
- Initialization tests:     ✓ 2/2 methods tested (100%)
- Metadata management:      ✓ ~2/2 methods tested (100%)
- Plotting methods:         ✗ 0/9 methods tested (0%)
- ROI/Geometry:            ✗ 0/2 methods tested (0%)
- Export operations:        ✗ 0/3 methods tested (0%)
- Data retrieval:          ✗ 0/4 methods tested (0%)
- Job management:          ✗ 0/4 methods tested (0%)
- Utility methods:         ✗ 0/2 methods tested (0%)

ACTUAL COVERAGE: ~4% of public methods
```

#### Recommended Priority Test Additions (P1)

**High Priority (Blocks Release):**
1. `test_plot_g2_with_valid_data()` - Basic G2 plotting
2. `test_plot_saxs_1d_with_valid_data()` - Basic SAXS 1D
3. `test_plot_saxs_2d_with_valid_data()` - SAXS 2D plotting
4. `test_add_roi_ring()` - Ring ROI addition
5. `test_add_roi_sector()` - Sector ROI addition
6. `test_export_g2_to_file()` - G2 export
7. `test_export_saxs_1d_to_file()` - SAXS 1D export
8. `test_plot_twotime_with_valid_data()` - TwoTime plotting

**Medium Priority:**
9. `test_plot_g2_stability()` - Stability analysis
10. `test_export_diffusion()` - Diffusion export
11. `test_get_info_at_mouse()` - Mouse information
12. `test_submit_job()` - Job submission

---

### 3. xpcsviewer/module/g2mod.py

**Status:** MODERATE - G2 analysis module with 11 public functions, 82 tests but ~35% coverage

#### Overview
- **File Path:** `/Users/b80985/Projects/xpcsviewer/xpcsviewer/module/g2mod.py`
- **Functions:** 11 public functions (module-level, not class-based)
- **Test Files:**
  - `/Users/b80985/Projects/xpcsviewer/tests/analysis/g2_analysis/test_g2mod.py` (58 tests)
  - `/Users/b80985/Projects/xpcsviewer/tests/analysis/g2_analysis/test_g2_analysis.py` (24 tests)
- **Total Tests:** 82
- **Estimated Coverage:** ~35%

#### Public Functions

**Well-Tested (5 of 11):**
- `get_data(xf_list, q_range=None, t_range=None)` - ✓ Tested
- `get_g2_stability_data(xf_obj, q_range=None, t_range=None)` - ✓ Tested
- `compute_geometry(g2, plot_type)` - ✓ Tested
- `pg_plot(ax, x, y, dy, color, label, symbol, symbol_size=5)` - ✓ Tested
- `vectorized_g2_baseline_correction(g2_data, baseline_values)` - ✓ Tested

**Partially Tested (3 of 11):**
- `pg_plot_stability(ax, ...)` - ✓ Some coverage
- `pg_plot_one_g2(ax, x, y, dy, color, label, symbol, symbol_size=5)` - ✓ Some coverage
- `vectorized_g2_interpolation(tel, g2_data, target_tel)` - ✓ Some coverage

**Untested (3 of 11):**
- `batch_g2_normalization(g2_data_list, method="max")` - **UNTESTED**
- `compute_g2_ensemble_statistics(g2_data_list)` - **UNTESTED**
- `optimize_g2_error_propagation(g2_data, g2_errors, operations)` - **UNTESTED**

#### Critical Gaps

1. **Batch Operations** - Multiple G2 normalization and statistics
2. **Error Propagation** - Error handling for data pipelines
3. **Integration Tests** - End-to-end G2 analysis workflows

#### Recommended Priority Additions (P1)

1. `test_batch_g2_normalization_max_method()`
2. `test_batch_g2_normalization_mean_method()`
3. `test_compute_g2_ensemble_statistics()`
4. `test_optimize_g2_error_propagation()`

---

### 4. xpcsviewer/module/twotime.py

**Status:** MODERATE - TwoTime analysis module with 4 public functions, 30 tests, ~30% coverage

#### Overview
- **File Path:** `/Users/b80985/Projects/xpcsviewer/xpcsviewer/module/twotime.py`
- **Functions:** 4 public functions
- **Test File:** `/Users/b80985/Projects/xpcsviewer/tests/analysis/twotime_analysis/test_twotime_analysis.py`
- **Test Count:** 30 tests
- **Estimated Coverage:** ~30%

#### Public Functions

**Tested (2 of 4):**
- `clean_c2_for_visualization(c2, method="nan_to_num")` - ✓ Tested
- `calculate_safe_levels(c2)` - ✓ Tested

**Untested (2 of 4):**
- `plot_twotime(hdl, c2_result)` - **UNTESTED**
- `plot_twotime_g2(hdl, c2_result)` - **UNTESTED**

#### Critical Gaps

1. **Visualization Methods** - TwoTime plotting functions untested
2. **Integration with GUI** - Handler interaction testing needed

#### Recommended Priority Additions (P2)

1. `test_plot_twotime_with_valid_c2()`
2. `test_plot_twotime_g2_with_valid_c2()`
3. `test_plot_twotime_error_handling()`

---

### 5. xpcsviewer/fileIO/hdf_reader.py

**Status:** GOOD - HDF5 reader with 25+ public methods, 85 tests, ~20% actual method coverage

#### Overview
- **File Path:** `/Users/b80985/Projects/xpcsviewer/xpcsviewer/fileIO/hdf_reader.py`
- **Classes:** PooledConnection, ConnectionPool
- **Test File:** `/Users/b80985/Projects/xpcsviewer/tests/unit/fileio/test_hdf_reader.py`
- **Test Count:** 85 tests
- **Estimated Coverage:** ~20% (tests focus on edge cases)

#### Analysis

**Well-Tested Components:**
- Connection pooling logic
- Health checking
- Error handling
- Cache management

**Coverage Gaps:**
- Some advanced query patterns
- Multi-file operations
- Performance optimization paths

#### Recommended Priority Additions (P2)

1. `test_batch_read_with_large_files()`
2. `test_connection_reuse_efficiency()`
3. `test_cache_eviction_strategy()`

---

### 6. xpcsviewer/fileIO/qmap_utils.py

**Status:** GOOD - Q-map utilities with 20+ methods, 50 tests, ~40% coverage

#### Overview
- **File Path:** `/Users/b80985/Projects/xpcsviewer/xpcsviewer/fileIO/qmap_utils.py`
- **Classes:** QMapManager, QMapIO
- **Test File:** `/Users/b80985/Projects/xpcsviewer/tests/unit/fileio/test_qmap_utils.py`
- **Test Count:** 50 tests
- **Estimated Coverage:** ~40%

#### Strengths

- Good coverage of QMapIO initialization
- Q-map computation testing
- Error handling for invalid inputs

#### Gaps

- Q-map caching efficiency
- Large file handling
- Performance optimization

---

### 7. xpcsviewer/module/saxs1d.py

**Status:** MODERATE - SAXS 1D analysis with 10 functions, 24 tests, ~40% coverage

#### Overview
- **File Path:** `/Users/b80985/Projects/xpcsviewer/xpcsviewer/module/saxs1d.py`
- **Functions:** 10 public functions
- **Test File:** `/Users/b80985/Projects/xpcsviewer/tests/analysis/saxs_analysis/test_saxs_analysis.py`
- **Test Count:** 24 tests
- **Estimated Coverage:** ~40%

#### Gaps

1. **ROI Extraction** - `optimize_roi_extraction()` untested
2. **Batch Operations** - `batch_saxs_analysis()` partially tested
3. **Integration Tests** - End-to-end workflows

---

## Cross-Cutting Issues

### 1. God Object Pattern

**Identified God Objects:**
- `XpcsFile` (39 public methods) - Combines:
  - Data loading (HDF5 I/O)
  - Data access (properties, getters)
  - Analysis (fitting, ROI extraction)
  - Export (SAXS, G2, diffusion)
  - Caching (memory management)

- `ViewerKernel` (28 public methods) - Combines:
  - Data orchestration
  - Plotting (9 plot methods)
  - Export (3 export methods)
  - Job management (4 methods)
  - State management

**Impact on Testing:**
- High complexity increases test brittleness
- Methods have unclear boundaries
- Changes ripple across test suite
- Difficult to isolate failures

**Recommendation:** Consider decomposition into:
- DataManager (I/O, loading, caching)
- AnalysisEngine (fitting, processing)
- ExportManager (file export)
- PlottingOrchestrator (visualization)

### 2. Missing Integration Tests

**Identified Gaps:**
1. No end-to-end G2 analysis tests (load → normalize → fit → export)
2. No ROI extraction pipeline tests
3. No TwoTime analysis workflow tests
4. No parallel processing validation

### 3. Missing Fixture Coverage

**Data Access Patterns Not Tested:**
- Large file handling (>1GB HDF5)
- Multiple ROI extraction
- Parallel fitting operations
- Memory pressure scenarios
- Cache eviction under load

### 4. Missing Edge Case Coverage

**Common Edge Cases Untested:**
- Empty Q-ranges
- Single-point datasets
- NaN/Inf handling
- Division by zero in normalization
- File handle exhaustion
- Concurrent access patterns

---

## Test Architecture Recommendations

### 1. Implement Test Pyramid for XpcsFile

```
     /\
    /  \  <- E2E tests (5%)
   /    \ - Full pipeline: load → analyze → export
  /      \
 /--------\  <- Integration tests (20%)
/          \ - Data loading + Analysis operations
/            \
/              \ <- Unit tests (75%)
/                \- Individual methods with mocks
```

### 2. Priority Test Suite Structure

```python
tests/
├── integration/
│   ├── g2_analysis_pipeline.py      # Load → Fit → Export
│   ├── twotime_analysis_pipeline.py # TwoTime workflow
│   ├── roi_extraction_pipeline.py   # ROI extraction
│   └── memory_stress_tests.py       # Memory management
├── unit/
│   ├── core/
│   │   ├── test_xpcs_file_data_access.py  # Data getters
│   │   ├── test_xpcs_file_fitting.py      # Fitting methods
│   │   ├── test_xpcs_file_export.py       # Export methods
│   │   ├── test_xpcs_file_roi.py          # ROI methods
│   │   ├── test_xpcs_file_cache.py        # Cache methods
│   │   ├── test_viewer_kernel_plotting.py # Plot methods
│   │   └── test_viewer_kernel_export.py   # Export methods
└── fixtures/
    ├── xpcs_sample_data.py   # Real HDF5 test data
    ├── xpcs_mock_data.py     # Synthetic data
    └── analysis_parameters.py # Test parameters
```

### 3. Test Data Strategy

**Current Gap:** Most tests use mocks, missing real HDF5 data

**Recommendation:**
```python
# Create fixtures for different analysis types
@pytest.fixture
def hdf5_multitau_file(tmp_path):
    """Real Multitau HDF5 file for testing."""
    # Generate or provide sample file
    return tmp_path / "multitau_sample.hdf5"

@pytest.fixture
def hdf5_twotime_file(tmp_path):
    """Real TwoTime HDF5 file for testing."""
    return tmp_path / "twotime_sample.hdf5"

@pytest.fixture
def hdf5_saxs_only_file(tmp_path):
    """SAXS-only HDF5 file for testing."""
    return tmp_path / "saxs_sample.hdf5"
```

---

## Coverage Target Roadmap

### Phase 1: Critical Paths (Sprint 1-2)

**Target: 50% coverage for XpcsFile and ViewerKernel**

Priority Methods to Test:
1. XpcsFile data access (get_g2_data, get_twotime_c2)
2. XpcsFile fitting operations (fit_g2, fit_tauq)
3. ViewerKernel plotting methods (plot_g2, plot_saxs_1d)
4. ViewerKernel export methods (export_g2, export_saxs_1d)

**Estimated Effort:** 40-50 test cases, ~60 hours

### Phase 2: Integration Tests (Sprint 3-4)

**Target: 30% integration test coverage**

End-to-End Workflows:
1. G2 Analysis Pipeline (load → normalize → fit → export)
2. SAXS ROI Extraction (single + parallel)
3. TwoTime Analysis (load → correct → visualize)
4. Memory Management (cache, cleanup, streaming)

**Estimated Effort:** 20-30 test cases, ~40 hours

### Phase 3: Edge Cases & Error Handling (Sprint 5-6)

**Target: 80%+ coverage with edge case handling**

Error Scenarios:
1. Corrupted HDF5 files
2. Missing data fields
3. Out-of-memory conditions
4. Concurrent access conflicts
5. Invalid parameter ranges

**Estimated Effort:** 30-40 test cases, ~50 hours

---

## Metrics & Monitoring

### Coverage Targets by Phase

| Phase | Module | Target | Current | Gap |
|-------|--------|--------|---------|-----|
| 1 | XpcsFile | 50% | 5% | +45% |
| 1 | ViewerKernel | 50% | 4% | +46% |
| 2 | Integration | 30% | 0% | +30% |
| 3 | Edge Cases | 20% | ~5% | +15% |

### Code Quality Gates

```python
# Required coverage thresholds
min_coverage = 0.70  # 70% minimum
min_unit_tests = 0.50  # 50% of tests must be unit
min_integration = 0.20  # 20% must be integration
max_mock_ratio = 0.80  # Max 80% mocked dependencies
```

### CI/CD Integration

```yaml
# GitHub Actions: test-coverage.yml
- name: Run Test Coverage
  run: pytest --cov=xpcsviewer --cov-report=xml

- name: Check Coverage Gates
  run: |
    python scripts/check_coverage_gates.py \
      --min-overall=70 \
      --min-file=65 \
      --fail-under=65
```

---

## Risk Assessment

### High Risk Areas

**1. XpcsFile.fit_g2() - No Unit Tests**
- Risk Level: CRITICAL
- Impact: Core analysis feature
- Workaround: Currently tested indirectly through GUI
- Recommendation: Add 5+ focused unit tests

**2. ViewerKernel.plot_*() Methods - No Tests**
- Risk Level: CRITICAL
- Impact: All visualization features
- Workaround: Manual GUI testing only
- Recommendation: Add 10+ visualization tests with assertion helpers

**3. ROI Extraction Pipeline - Limited Tests**
- Risk Level: HIGH
- Impact: Custom analysis workflows
- Workaround: Single simple ROI tested
- Recommendation: Add parallel extraction, error handling, performance tests

**4. TwoTime Visualization - No Tests**
- Risk Level: MEDIUM
- Impact: TwoTime analysis feature
- Workaround: Visual inspection only
- Recommendation: Add 2-3 visualization tests

### Medium Risk Areas

- Cache management under memory pressure
- Batch export operations
- Multi-file processing
- Error propagation in pipelines

---

## Appendix: File Locations Quick Reference

```
Core Legacy Modules:
├── xpcsviewer/xpcs_file.py              (80 items, 39 public methods)
├── xpcsviewer/viewer_kernel.py          (28 public methods)
│
Module Analysis:
├── xpcsviewer/module/g2mod.py           (11 functions)
├── xpcsviewer/module/twotime.py         (4 functions)
├── xpcsviewer/module/saxs1d.py          (10 functions)
├── xpcsviewer/module/saxs2d.py          (1 function)
├── xpcsviewer/module/tauq.py            (2 functions)
├── xpcsviewer/module/intt.py            (4 functions)
│
FileIO Modules:
├── xpcsviewer/fileIO/hdf_reader.py      (25+ methods)
├── xpcsviewer/fileIO/qmap_utils.py      (20+ methods)
├── xpcsviewer/fileIO/ftype_utils.py     (3 functions)
├── xpcsviewer/fileIO/aps_8idi.py        (specific format handling)
│
Test Files:
├── tests/unit/core/test_xpcs_file.py    (39 tests)
├── tests/unit/core/test_viewer_kernel.py (50 tests)
├── tests/analysis/g2_analysis/          (82 tests total)
├── tests/analysis/twotime_analysis/     (30 tests)
├── tests/analysis/saxs_analysis/        (24 tests)
└── tests/unit/fileio/                   (135+ tests)
```

---

## Summary & Next Steps

### Key Findings

1. **Critical Coverage Gaps in God Objects**
   - XpcsFile: 39 public methods, ~5% tested
   - ViewerKernel: 28 public methods, ~4% tested

2. **Missing End-to-End Tests**
   - No integration tests for complete analysis pipelines
   - Manual testing is primary validation method

3. **Uneven Coverage Across Modules**
   - FileIO well-tested (85+ tests)
   - G2 analysis moderate (82 tests, 35% coverage)
   - Core analysis orchestration almost untested (50 tests, 5% actual)

4. **Architecture Debt**
   - God object pattern makes testing difficult
   - High coupling between components

### Immediate Actions (This Sprint)

1. Create test specification for XpcsFile critical paths
2. Create test specification for ViewerKernel critical paths
3. Set up test data generation utilities
4. Create assertion helpers for visualization testing
5. Begin Phase 1 implementation (critical paths)

### Long-Term Recommendations

1. Refactor god objects into focused components
2. Implement integration test framework
3. Add performance regression testing
4. Establish continuous coverage monitoring
5. Implement mutation testing to find weak assertions

---

**Report Generated:** 2026-01-06
**Analysis Scope:** Legacy component test coverage evaluation
**Prepared For:** Test Automation & Quality Engineering Review
