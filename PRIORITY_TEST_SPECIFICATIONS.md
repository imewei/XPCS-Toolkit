# Priority Test Specifications - XPCS Viewer

**Document Type:** Test Specification
**Priority:** P1 - Critical Path Tests
**Sprint:** Phase 1 (Blocks Release)
**Target Modules:** XpcsFile, ViewerKernel

---

## Test Suite 1: XpcsFile Data Access Methods

### Test Group: G2 Data Access

**Test 1.1: Get G2 Data - Basic Functionality**
```
Test ID: XF-001
Module: xpcsviewer.xpcs_file.XpcsFile
Method: get_g2_data(qrange=None, trange=None)
Status: UNTESTED

Objective:
  Verify that get_g2_data() correctly retrieves G2 correlation data
  for single Q-range and time-range selection.

Test Data:
  - Input: Multitau analysis HDF5 file with complete G2 dataset
  - Q-range: (0.001, 0.1) Angstrom^-1
  - T-range: (0, 100) indices

Expected Behavior:
  - Returns numpy array with shape (time_points, 3)
    [delay_time, g2_value, g2_error]
  - Values within physical constraints (g2 >= 1.0)
  - No NaN/Inf values
  - Correct Q-bin selected

Assertions:
  assert g2_data.shape == (n_time_points, 3)
  assert np.all(g2_data[:, 1] >= 1.0)  # g2 >= 1
  assert np.all(np.isfinite(g2_data))   # No NaN/Inf
  assert result['q_selected'] == qbin_index

Error Cases:
  - Invalid Q-range (None) → Use default
  - Invalid T-range (None) → Use all
  - Empty Q-range → Raise ValueError
  - Non-existent Q-bin → Raise KeyError

Execution: ~100ms
```

**Test 1.2: Get G2 Data - Multiple Q-Ranges**
```
Test ID: XF-002
Method: get_g2_data(qrange, trange)

Objective:
  Verify G2 data retrieval for multiple Q-range selections.

Test Cases:
  a) Small Q-range: (0.001, 0.005)
  b) Medium Q-range: (0.05, 0.1)
  c) Large Q-range: (0.001, 0.15)
  d) Single Q-value as range: (0.05, 0.05)

Expected:
  - Each case returns valid G2 data
  - Data consistency across ranges
  - Q-bin mapping correct

Assertions:
  for each test case:
    assert result is not None
    assert result['q_bin'] in valid_qbin_range
    assert result['g2'].min() >= 1.0
```

**Test 1.3: Get G2 Data - Partial Time Range**
```
Test ID: XF-003
Method: get_g2_data(qrange, trange)

Objective:
  Verify time-range slicing works correctly.

Test Cases:
  a) First 10 points: (0, 10)
  b) Middle section: (50, 70)
  c) Last 10 points: (n-10, n)
  d) Single point: (25, 25)

Expected:
  - Returned data respects time-range limits
  - Proper indexing (inclusive both ends)
  - Correct row count

Assertions:
  assert len(result) == (t_range[1] - t_range[0] + 1)
  assert result.index[0] == tau_values[t_range[0]]
  assert result.index[-1] == tau_values[t_range[1]]
```

---

### Test Group: G2 Stability Analysis

**Test 1.4: Get G2 Stability Data - Basic**
```
Test ID: XF-004
Module: xpcsviewer.xpcs_file.XpcsFile
Method: get_g2_stability_data(xf_obj, qrange=None, trange=None)
Status: UNTESTED

Objective:
  Verify G2 stability analysis retrieves multi-frame G2 evolution.

Test Data:
  - Multitau file with 50+ frames
  - Single Q-range selection

Expected:
  - Returns dict with keys: ['frame_indices', 'g2_values', 'g2_errors']
  - Shape: (n_frames, n_time_points)
  - Temporal progression visible

Assertions:
  assert 'frame_indices' in result
  assert result['g2_values'].shape == (n_frames, n_time_points)
  assert np.all(result['g2_values'] >= 1.0)
```

---

### Test Group: TwoTime Correlation Data

**Test 1.5: Get TwoTime C2 Matrix - Basic**
```
Test ID: XF-005
Module: xpcsviewer.xpcs_file.XpcsFile
Method: get_twotime_c2(selection=0, correct_diag=True, max_size=32678)
Status: UNTESTED

Objective:
  Verify TwoTime C2 matrix retrieval with diagonal correction.

Test Data:
  - TwoTime analysis HDF5 file
  - Single selection index

Expected:
  - Returns 2D C2 matrix (t1, t2)
  - Shape: (delay_points, delay_points) or smaller if max_size enforced
  - Diagonal corrected if flag=True
  - Symmetric or near-symmetric

Assertions:
  assert c2.ndim == 2
  assert c2.shape[0] <= max_size
  assert np.all(np.isfinite(c2))
  # Check symmetry: abs(c2 - c2.T) < threshold
  assert np.allclose(c2, c2.T, atol=1e-6, rtol=1e-4)
```

**Test 1.6: Get TwoTime C2 - Diagonal Correction Impact**
```
Test ID: XF-006
Method: get_twotime_c2(..., correct_diag=True/False)

Objective:
  Verify diagonal correction changes values as expected.

Test Cases:
  a) With correction: correct_diag=True
  b) Without correction: correct_diag=False

Expected:
  - With correction: diagonal values differ from uncorrected
  - Diagonal values in corrected version typically larger
  - Off-diagonal unaffected

Assertions:
  c2_corrected = get_twotime_c2(..., correct_diag=True)
  c2_uncorrected = get_twotime_c2(..., correct_diag=False)

  # Diagonal differs
  assert not np.allclose(np.diag(c2_corrected),
                         np.diag(c2_uncorrected))
  # Off-diagonal same
  assert np.allclose(c2_corrected - np.diag(np.diag(c2_corrected)),
                     c2_uncorrected - np.diag(np.diag(c2_uncorrected)))
```

---

### Test Group: ROI Data Extraction

**Test 1.7: Get ROI Data - Single ROI**
```
Test ID: XF-007
Module: xpcsviewer.xpcs_file.XpcsFile
Method: get_roi_data(roi_parameter, phi_num=180)
Status: UNTESTED

Objective:
  Verify single ROI extraction and phi binning.

Test Data:
  - SAXS 2D image with ring structure
  - ROI parameters: center=(512, 512), q_range=(0.01, 0.05)
  - phi_num=180 (divide azimuth into 180 bins)

Expected:
  - Returns dict with:
    - 'qbin': selected q-bin index
    - 'data': SAXS 1D profile (I vs phi)
    - 'phi': azimuthal angles
  - Shape: (phi_num,)
  - No NaN values

Assertions:
  assert 'qbin' in result
  assert 'data' in result
  assert 'phi' in result
  assert len(result['data']) == phi_num
  assert np.all(np.isfinite(result['data']))
```

**Test 1.8: Get ROI Data - Parallel Multiple ROIs**
```
Test ID: XF-008
Module: xpcsviewer.xpcs_file.XpcsFile
Method: get_multiple_roi_data_parallel(roi_list, phi_num=180, max_workers=None)
Status: UNTESTED

Objective:
  Verify parallel ROI extraction efficiency and correctness.

Test Data:
  - List of 5 ROI parameter sets
  - SAXS 2D image 2048x2048
  - max_workers=4

Expected:
  - Returns list of 5 ROI results
  - Same results as sequential extraction
  - Parallel execution faster than sequential

Assertions:
  results = get_multiple_roi_data_parallel(roi_list, max_workers=4)

  assert len(results) == len(roi_list)
  assert all(isinstance(r, dict) for r in results)
  assert all('data' in r for r in results)

  # Verify consistency with single extraction
  for i, roi in enumerate(roi_list):
    single_result = get_roi_data(roi)
    assert np.allclose(results[i]['data'], single_result['data'])
```

---

## Test Suite 2: XpcsFile Analysis Methods

### Test Group: G2 Fitting

**Test 2.1: Fit G2 - Single Exponential**
```
Test ID: XF-009
Module: xpcsviewer.xpcs_file.XpcsFile
Method: fit_g2(...) with single exponential model
Status: UNTESTED

Objective:
  Verify single-exponential fitting produces correct relaxation time.

Test Data:
  - Synthetic G2 data: g2(t) = 1 + A*exp(-2*t/tau)
  - Known tau=10ms, A=0.5, baseline=1.0
  - 50 time points with small random noise

Expected:
  - Fitted tau ≈ 10ms (within 10%)
  - Fitted amplitude ≈ 0.5 (within 10%)
  - Reduced chi-squared < 2
  - Convergence achieved

Assertions:
  fit_result = xfile.fit_g2(tau, g2_data, 'single')

  assert abs(fit_result['tau'] - 10.0) < 1.0  # 10% tolerance
  assert abs(fit_result['amplitude'] - 0.5) < 0.05
  assert fit_result['chi_squared'] < 2.0
  assert fit_result['success'] == True
```

**Test 2.2: Fit G2 - Double Exponential**
```
Test ID: XF-010
Method: fit_g2(...) with double exponential model

Objective:
  Verify double-exponential fitting for bimodal dynamics.

Test Data:
  - Synthetic: g2(t) = 1 + A1*exp(-2*t/tau1) + A2*exp(-2*t/tau2)
  - tau1=5ms, tau2=50ms
  - A1=0.3, A2=0.2

Expected:
  - Fitted tau1, tau2 within 10% of true values
  - Fitted amplitudes correct
  - Baseline at 1.0

Assertions:
  fit_result = xfile.fit_g2(tau, g2_data, 'double')

  # Check tau1 (faster process)
  assert abs(fit_result['tau1'] - 5.0) < 0.5
  assert abs(fit_result['tau2'] - 50.0) < 5.0
  assert abs(fit_result['amplitude1'] - 0.3) < 0.03
  assert abs(fit_result['amplitude2'] - 0.2) < 0.02
```

**Test 2.3: Fit G2 - Error Handling**
```
Test ID: XF-011
Method: fit_g2(...) error cases

Objective:
  Verify fitting handles bad data gracefully.

Test Cases:
  a) All NaN data
  b) Constant (flat) data
  c) Negative g2 values (invalid)
  d) Empty array
  e) Too few points (< 3)

Expected:
  - Each case raises appropriate exception
  - Clear error message
  - No silent failures

Assertions:
  with pytest.raises(ValueError, match="invalid data"):
    xfile.fit_g2(tau, nan_data)

  with pytest.raises(ValueError, match="g2 must be >= 1"):
    xfile.fit_g2(tau, negative_g2)
```

---

### Test Group: Tau-Q Analysis

**Test 2.4: Fit Tau-Q - Power Law**
```
Test ID: XF-012
Module: xpcsviewer.xpcs_file.XpcsFile
Method: fit_tauq(q_range, bounds, fit_flag, force_refit=False)
Status: UNTESTED

Objective:
  Verify Q-dependent relaxation time fitting.

Test Data:
  - Multiple Q-bins with fitted tau values
  - Q-range: [0.01, 0.02, 0.05, 0.1] Angstrom^-1
  - Expected: tau(q) = tau0 * q^(-alpha)
  - Known: tau0=1ms, alpha=2.0

Expected:
  - Fitted tau0 and alpha within bounds
  - Power-law relationship verified
  - Plot generated

Assertions:
  result = xfile.fit_tauq(q_range=(0.01, 0.1),
                          bounds={'tau0': (0.1, 10), 'alpha': (1, 3)},
                          fit_flag=True)

  assert abs(result['tau0'] - 1.0) < 0.1
  assert abs(result['alpha'] - 2.0) < 0.1
  assert result['fit_success'] == True
```

---

## Test Suite 3: ViewerKernel Plotting Methods

### Test Group: G2 Plotting

**Test 3.1: Plot G2 - Valid Data**
```
Test ID: VK-001
Module: xpcsviewer.viewer_kernel.ViewerKernel
Method: plot_g2(handler, q_range, t_range, y_range, rows=None, **kwargs)
Status: UNTESTED

Objective:
  Verify G2 plot generation with correct data representation.

Test Data:
  - Mock XpcsFile with g2 data
  - Mock matplotlib handler
  - Q-range: (0.01, 0.1), T-range: (0, 100)

Expected:
  - Handler receives plot data
  - Axes labeled correctly
  - Legend present
  - Error bars shown

Assertions:
  mock_handler = Mock()
  vk.plot_g2(mock_handler, (0.01, 0.1), (0, 100), None)

  # Verify handler methods called
  mock_handler.plot.assert_called()
  mock_handler.set_xlabel.assert_called_with('Delay Time (s)')
  mock_handler.set_ylabel.assert_called_with('g2(t)')

  # Verify legend added
  assert mock_handler.legend.called or 'g2' in mock_handler.plot.call_args
```

**Test 3.2: Plot G2 - Multiple Q-Ranges (Overlay)**
```
Test ID: VK-002
Method: plot_g2(...) with multiple selections

Objective:
  Verify plotting multiple G2 curves on same axes.

Test Cases:
  a) Single Q-bin
  b) Multiple Q-bins as overlay
  c) Comparison of different samples

Expected:
  - Each Q-bin different color
  - Legend identifies each curve
  - Y-axis auto-scaled appropriately
```

---

### Test Group: SAXS Plotting

**Test 3.3: Plot SAXS 1D - Valid Data**
```
Test ID: VK-003
Module: xpcsviewer.viewer_kernel.ViewerKernel
Method: plot_saxs_1d(pg_hdl, mp_hdl, **kwargs)
Status: UNTESTED

Objective:
  Verify SAXS 1D plot in PyQtGraph handler.

Test Data:
  - Mock PyQtGraph image view
  - Mock matplotlib handler for annotations
  - SAXS 1D profile data

Expected:
  - Image displayed in PyQtGraph
  - Matplotlib annotations added
  - Color scale visible

Assertions:
  mock_pg = Mock()
  mock_mp = Mock()

  vk.plot_saxs_1d(mock_pg, mock_mp)

  mock_pg.setImage.assert_called()
  mock_mp.plot.assert_called()
```

**Test 3.4: Plot SAXS 2D - with ROI**
```
Test ID: VK-004
Method: plot_saxs_2d(*args, rows=None, **kwargs)

Objective:
  Verify SAXS 2D detector image with ROI overlay.

Test Data:
  - 2D detector image
  - ROI parameters (ring or sector)

Expected:
  - Image displayed
  - ROI geometry drawn
  - Interactive selection possible
```

---

### Test Group: TwoTime Visualization

**Test 3.5: Plot TwoTime Heatmap**
```
Test ID: VK-005
Module: xpcsviewer.viewer_kernel.ViewerKernel
Method: plot_twotime(hdl, rows=None, **kwargs)
Status: UNTESTED

Objective:
  Verify TwoTime heatmap visualization.

Test Data:
  - C2 matrix (symmetric, 100x100)
  - Time delay values

Expected:
  - Heatmap displayed
  - Color scale represents C2 magnitude
  - Axes labeled with delay times
  - NaN values handled (masked)

Assertions:
  mock_hdl = Mock()

  vk.plot_twotime(mock_hdl)

  mock_hdl.setImage.assert_called()
  call_args = mock_hdl.setImage.call_args

  # Verify image data is 2D
  assert len(call_args[0][0].shape) == 2
  # Verify no NaN in displayed data
  assert np.all(np.isfinite(call_args[0][0]))
```

---

### Test Group: Export Operations

**Test 3.6: Export G2 Data - ASCII Format**
```
Test ID: VK-006
Module: xpcsviewer.viewer_kernel.ViewerKernel
Method: export_g2(folder, rows=None)
Status: UNTESTED

Objective:
  Verify G2 data export to ASCII file.

Test Data:
  - G2 data for 3 Q-bins
  - Temporary folder

Expected:
  - Creates file: g2_export.txt or similar
  - Columns: tau, g2, g2_error
  - Proper formatting
  - Metadata in header

Assertions:
  tmp_folder = tmpdir
  vk.export_g2(str(tmp_folder))

  export_file = list(tmp_folder.glob('g2*.txt'))[0]
  assert export_file.exists()

  data = np.loadtxt(export_file, skiprows=10)
  assert data.shape[1] == 3  # tau, g2, g2_err
  assert np.all(data[:, 1] >= 1.0)  # g2 valid
```

**Test 3.7: Export SAXS 1D - Multiple ROI**
```
Test ID: VK-007
Method: export_saxs_1d(pg_hdl, folder)

Objective:
  Verify SAXS 1D export for multiple ROI selections.

Expected:
  - One file per ROI
  - File naming: saxs_roi_000.txt, saxs_roi_001.txt, ...
  - Format: q, I(q), error
  - Header with ROI parameters

Assertions:
  vk.export_saxs_1d(mock_pg, tmp_folder)

  files = list(tmp_folder.glob('saxs_roi_*.txt'))
  assert len(files) == num_rois

  for f in files:
    data = np.loadtxt(f, skiprows=5)
    assert data.shape[1] == 3  # q, I, error
```

---

### Test Group: ROI Operations

**Test 3.8: Add ROI - Ring Geometry**
```
Test ID: VK-008
Module: xpcsviewer.viewer_kernel.ViewerKernel
Method: add_roi(hdl, **kwargs) with ring ROI
Status: UNTESTED

Objective:
  Verify ring ROI addition to detector image.

Test Data:
  - Mock image handler
  - Ring parameters: center=(512,512), r_inner=100, r_outer=150

Expected:
  - Ring ROI drawn on image
  - Interactive handles visible
  - Parameter extraction works

Assertions:
  mock_hdl = Mock()

  vk.add_roi(mock_hdl, roi_type='ring', center=(512,512),
             r_inner=100, r_outer=150)

  # Verify ROI drawn
  assert mock_hdl.addItem.called or mock_hdl.plot.called
```

**Test 3.9: Add ROI - Sector Geometry**
```
Test ID: VK-009
Method: add_roi(hdl, **kwargs) with sector ROI

Objective:
  Verify sector/wedge ROI addition.

Test Data:
  - Sector parameters: center=(512,512), r=100,
    angle_start=0, angle_end=45 degrees

Expected:
  - Wedge-shaped ROI drawn
  - Angle selection interactive
```

---

## Test Suite 4: Edge Cases & Error Handling

### Test Group: Invalid Input Handling

**Test 4.1: XpcsFile - Invalid Q-Range**
```
Test ID: ERR-001
Scenario: User provides invalid Q-range to get_g2_data()

Test Cases:
  a) Negative Q values: (-0.1, 0.05) → ValueError
  b) Inverted range: (0.1, 0.01) → ValueError or swap?
  c) Out of bounds: (1.0, 2.0) when max_q=0.5 → ValueError
  d) NaN values: (nan, 0.1) → ValueError

Expected:
  - Clear error message
  - No silent failures
  - Helpful suggestions

Assertions:
  with pytest.raises(ValueError, match="Q range out of bounds"):
    xfile.get_g2_data(qrange=(1.0, 2.0))
```

### Test Group: Memory & Performance

**Test 4.2: Large File Memory Handling**
```
Test ID: PERF-001
Scenario: Load large HDF5 file (>1GB)

Expected:
  - Lazy loading works
  - Memory usage bounded
  - Streaming data access
  - No OOM crash

Assertions:
  initial_mem = get_memory_usage()

  # Access large dataset
  result = xfile.get_twotime_stream(max_size=10000)

  peak_mem = get_memory_usage()

  # Memory increase < 500MB
  assert (peak_mem - initial_mem) < 500
```

---

## Test Execution Matrix

```
Test ID    Module      Function              Coverage  Effort  Status
──────────────────────────────────────────────────────────────────────
XF-001    XpcsFile    get_g2_data()         +1        1h      NEW
XF-002    XpcsFile    get_g2_data()         +1        1h      NEW
XF-003    XpcsFile    get_g2_data()         +1        1h      NEW
XF-004    XpcsFile    get_g2_stability()    +1        1h      NEW
XF-005    XpcsFile    get_twotime_c2()      +1        1.5h    NEW
XF-006    XpcsFile    get_twotime_c2()      +1        1h      NEW
XF-007    XpcsFile    get_roi_data()        +1        1h      NEW
XF-008    XpcsFile    get_multiple_roi()    +1        1.5h    NEW
XF-009    XpcsFile    fit_g2()              +1        2h      NEW
XF-010    XpcsFile    fit_g2()              +1        2h      NEW
XF-011    XpcsFile    fit_g2()              +1        1.5h    NEW
XF-012    XpcsFile    fit_tauq()            +1        2h      NEW
──────────────────────────────────────────────────────────────────────
VK-001    ViewerKernel plot_g2()            +1        1.5h    NEW
VK-002    ViewerKernel plot_g2()            +1        1h      NEW
VK-003    ViewerKernel plot_saxs_1d()       +1        1.5h    NEW
VK-004    ViewerKernel plot_saxs_2d()       +1        1.5h    NEW
VK-005    ViewerKernel plot_twotime()       +1        2h      NEW
VK-006    ViewerKernel export_g2()          +1        1.5h    NEW
VK-007    ViewerKernel export_saxs_1d()     +1        1.5h    NEW
VK-008    ViewerKernel add_roi()            +1        1.5h    NEW
VK-009    ViewerKernel add_roi()            +1        1h      NEW
──────────────────────────────────────────────────────────────────────
ERR-001   Multiple   Error handling        +2        2h      NEW
PERF-001  Multiple   Memory/Performance    +1        3h      NEW
──────────────────────────────────────────────────────────────────────

Phase 1 Totals:  21 tests, ~40 hours, +30 methods covered
```

---

## Success Criteria

**All Phase 1 tests must:**

1. Execute without failure
2. Have >90% assertion pass rate
3. Complete in <100ms each (except perf tests)
4. Include docstring with:
   - Test objective
   - Input data description
   - Expected behavior
   - Error cases (if applicable)
5. Use descriptive assertion messages
6. Handle cleanup/fixtures properly

**Coverage achieved:**
- XpcsFile: 5% → 50% (target)
- ViewerKernel: 4% → 50% (target)
- Overall: 18% → 45% (target)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-06
**Author:** Test Automation Team
**Status:** Ready for Implementation
