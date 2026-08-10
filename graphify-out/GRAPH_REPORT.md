# Graph Report - xpcsviewer  (2026-08-10)

## Corpus Check
- 361 files · ~613,924 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 11819 nodes · 18794 edges · 723 communities (466 shown, 257 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 1345 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `abb95d4b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- plot_posterior_predictive
- AsyncViewerKernel
- FR-014: Bayesian fit visualization with credible intervals
- .robust_curve_fit
- NLSQResult
- nlsq_optimize
- backend parametrized fixture
- ViewerKernel
- XpcsViewer
- FitResult
- TestMaskExport
- SessionManager
- xpcs_file.py
- MATPLOTLIB_DARK
- ._refresh_display
- BaseAsyncWorker
- TestRecentPathsManager
- xpcs_viewer.py
- Q-map pixel index fix: x/y are absolute indices 0..N-1 not beam-center offsets
- XPCS_USE_JAX
- tests/conftest.py
- QMapSchema
- IEEE 754 float precision fix: round xmap and bin edges to 12 decimal places before digitize
- JAXBackend
- BUG-006 HDF5 g2 shape axis
- XpcsFile
- run_tests
- MaskAssemble
- BUG-003 NLSQ sentinel result
- FileLocator
- ROIParameters
- SimpleMaskKernel
- process_c2_batch
- compute_transmission_qmap
- BaseAsyncWorker
- state_validator.py
- _get_module
- FR-013: generate_arviz_diagnostics with 6 standard plots
- .is_memory_pressure_high
- single_exp_func
- set_backend
- UnifiedMemoryManager
- double_exp
- Ui_mainWindow
- HDF5ConnectionPool
- xpcsviewer/simplemask/__init__.py
- LineROI
- QHBoxLayout
- .create_dataset
- ci_integration.py
- CoverageManager
- temporary_xpcs_file
- patch
- RateLimitedLogger
- benchmark
- qt_fixtures.py
- XPCSBaseError
- Numerical accuracy test suite: 62 passed, 5 skipped
- theme/__init__.py
- TestPartitionBlemishExport
- TestDataSpec
- XPCS Viewer (xpcsviewer) Python Package
- plot_nlsq_fit
- .validate_signal_connection
- xpcsviewer/utils/reliability.py
- TestBottleneck1G2EnsembleStatistics
- plot (tau-q)
- TestFileIOErrors
- TestCircleScaleHandles
- isolation.py
- .get_selected_rows
- 2.5 Threading Types (`xpcsviewer/threading/`)
- ConnectionStats
- MemoryMonitor
- ToastNotification
- BatchBayesianCoordinator
- FR-016: Prediction uncertainty via Jacobian/variance propagation
- qt_compat
- get_backend
- Contract Audit — Phase 2: Type and Contract Verification
- safe_json_write
- test_xpcs_file_data_access.py
- Integration Points Catalog
- XPCS Toolkit GUI Interactive Tests
- QtTestRunner
- gui
- LoggingContext
- BenchmarkTimer
- create_xpcs_dataset
- slice
- create_synthetic_g2_data
- .create_dataset
- test_twotime_qbin_memory.py
- gui
- BUG-F: tauq fit line render
- compute_uncertainty_band
- save_figure
- ThemeManager
- test_viewer_kernel_export.py
- ADR-003 HDF5 Facade
- Facade and Schema Infrastructure
- plot_comparison
- _cprofile_hotpaths.py
- SyntheticXPCSGenerator
- .load_path
- XPCS Toolkit Error Handling & Edge Case Test Suite
- test_viewer_kernel_plotting.py
- legacy.py
- ArrayType
- get_backend
- gui
- logging_config.py (LoggingConfig, get_logger, initialize_logging)
- RLock
- test_xpcs_file_fitting.py
- PooledConnection
- style_helpers.py
- SimpleMaskWindow
- gui_interactive/conftest.py
- Scientific Algorithm Validation Framework
- test_viewer_kernel.py
- .__init__
- Threading and Reliability Audit Report
- create_mock_hdf5_file
- test_xpcs_file_roi.py
- test_nlsq_bayesian_integration.py
- BackendProtocol
- backends/__init__.py
- nlsq.curve_fit (external)
- generate_arviz_diagnostics
- FR-008: Numerical accuracy across devices
- xpcsviewer.fitting
- QMapManager
- JSONFormatter
- ndarray
- fluerasu_2007_twotime (literature reference)
- LazyHDF5Array
- ._disconnect_signals
- XPCS Viewer — Architecture Map
- test_hdf5_facade.py
- ToastWidget
- validate_array_compatibility
- FR-021: validate_pcov for finite values and positive semi-definiteness
- layout_helpers.py
- Performance Optimization Summary Report
- Decision
- BUG-022: double-exp warm-start sorts tau
- TestSAXSVectorizedOperations
- TestToastManager
- MaskAssemble
- TestG2PartialSafetyCheck
- QMap
- jax_migration/conftest.py
- test_qt_jax_interop.py
- test_angular_computations.py
- test_float32_vs_float64.py
- get_icon
- NumPyBackend
- isolated_test_environment
- ndarray
- get_memory_manager
- .add_drawing
- JAX Backend Audit Report
- 2. P1 — High: Observable Wrong Behaviour / Reliability
- single_exp
- visualization.py
- TestGIXPCSPrecisionFormatting
- ThreadingViolationDetector
- measure_memory
- scientific_fixtures.py
- TestThemeSwitchingIntegration
- Algorithmic Bottleneck Analysis
- test_user_defined_gradients.py
- TestBackendDetection
- test_hdf5_jax_io.py
- TestPlotConstants
- h5py_mocks.py
- TestC2StatisticsBaseline
- AnalyticalBenchmarkSuite
- TestInitAverageSaveNamePreservation
- module.saxs1d
- test_calibration_baseline.py
- GUITestRunner
- test_memory_limits.py
- mathematical_invariants.py
- TestQmapColormapUIWidget
- TestG2MathematicalProperties
- TestCPUGPUNumericalEquivalence
- patch
- TestQMapConstants
- XPCS Toolkit Test Suite
- TestBottleneck5NLSQMultiStart
- ndarray
- JAX_PLATFORMS
- run_gui_tests.py
- TestBeamCenterCalibration
- T055: JIT compilation warmup test
- assemble_fit_summary
- Baseline Performance Profile Report
- Numerical and JAX Audit Report
- MplCanvas
- CleanupScheduler
- SmartFallbackManager
- TestTwoTimeCorrelationProperties
- TestSaxsBinningBaseline
- batch_read_fields
- TestSingletonDoubleCheckedLocking
- HDF5ValidationError
- xpcsviewer.utils.reliability
- gui
- TestQmapGradients
- constants.py
- physical_constraints.py
- ToastType
- E2KCONST = 12.398 keV Angstrom
- Any
- setter
- test_saxs_analysis.py
- reference_data/__init__.py
- test_simplemask_window.py
- test_simplemask_integration.py
- test_backend_detection.py
- TestDrawingToolsDefinitions
- test_qmap_integration.py
- TestPlotQmapColormapIntegration
- .create_xpcs_file
- Bayesian Fitting with NumPyro NUTS
- ImageViewDev
- test_bottleneck_analysis.py
- TestQmapBaseline
- XpcsViewer._collect_session_state
- TestDoubleExpBaseline
- T082: Float32 vs float64 precision test
- statistical_properties.py
- 3. P2 — Medium: Technical Debt / Maintainability
- test_bayesian_assembly.py
- export_bayesian_csv
- CommandPalette
- TestBug023FailedNLSQNaN
- ndimage.py
- TestQtTimerThreadingErrors
- _create_hdf5_structure
- TestBottleneck4C2Percentile
- test_cpu_only_launch.py
- test_plot_themes.py
- RecentPathsManager
- MockH5pyGroup
- ToastManager
- TestTwoTimeMatrixOperations
- .run_comprehensive_validation
- test_hotpath_baseline.py
- TestJITWarmup
- Python-Level Optimization Report
- .on_async_plot_ready
- SoftwarePackageValidator
- tests.scientific (package)
- ShortcutManager
- Subsystem Responsibilities
- test_g2_partial_safety.py
- test_gixpcs_precision.py
- Any
- XPCS Viewer Dependency Diagram
- create_slice
- ListDataModel
- MockH5py
- TestAPS8IDIPathFormats
- QtThreadingValidator
- ADR-003: HDF5 Facade Pattern with Connection Pooling
- TestLogTiming
- TAB_INDEX_CATEGORY
- qt_threading_utils.py
- LazyMplCanvasBarV
- Reference Data for Scientific Validation
- get_project_root
- framework/utils.py
- TestFittingAlgorithmProperties
- plot_bayesian_all_q
- test_tab_availability.py
- MemoryTestUtils
- .load_dataset
- ProgressDialog
- ProgressManager
- TestGetData
- TestComputeGeometry
- PerformanceTimer
- ComprehensiveCrossValidationFramework
- test_bayesian_dual_storage.py
- TestBug025FitDiagnosticsValidation
- .update_tab_availability
- ADR-002: Migration from scipy.optimize to NLSQ 0.6.0
- CommandAction
- XPCS Viewer Documentation Structure
- TestDragDropListViewMoveItem
- tests/utils/reliability.py
- vectorized_background_subtraction
- 1. P0 — Critical: Crash, Data Corruption, or Silent Wrong Result
- TestPlotThemesModule
- test_qss_lint.py
- test_session_field_completeness.py
- TestShortcutRegistration
- test_tg3_mask_export_and_g2_plot.py
- TestThemeManagerTokenAccess
- Backend Abstraction Pattern
- Data Flow
- TestGetShortcutMap
- test_partition.py
- TestEnsureNumpyAtPyQtGraphBoundaries
- .capture_qt_warnings
- ndarray
- ObjectRegistry
- HealthStatus
- ._monitoring_loop
- TestPgPlotFunction
- .evaluate_benchmark
- .run_comprehensive_validation
- LiteratureReferenceValidator
- TestRecentPathsManagerAddPath
- test_qt_error_detection.py
- TestSaxsBaseline
- MockH5pyFile
- TestSimpleMaskFromViewer
- tests.unit (package)
- test_qt_compat.py
- TestQtCompatLayer
- test_fitting_algorithms.py
- cross_validation_framework.py
- BUG-026: Legacy model factory get_backend once
- BUG-004 tau clamping NaN
- TestCommandPaletteSearch
- TestCommandPaletteInit
- test_theme_manager.py
- ._on_item_activated
- XPCS Viewer — Master Fix List
- .validate_fitting_algorithms
- TestThemeManagerBasics
- test_interpolation.py
- constants.__init__
- TestEraserTool
- XPCS Viewer Dependency Analysis and Integration Catalog
- 5.1 Critical Facades Needed
- DrawingTool
- TestGIXPCSScientificValidation
- BackgroundThreadTester
- TestEnvironmentValidator
- constants/__init__.py
- HealthMetric
- TestG2VectorizedOperations
- ScientificAssertions
- gui
- MockH5pyFile
- TestNoPySide6DirectImports
- test_package_basics.py
- MockQtEnvironment
- TestPerformanceMonitor
- take_snapshot
- ._connect_signals
- .get_health_summary
- HealthMonitor
- TestModuleIntegration
- TestPartitionMemoryEfficiency
- TestFileLoading
- fixture
- TestBackendArrayCreation
- TestSpecificFittingModels
- TestViewerKernelAverageWorker
- TestAPS8IDIKeyStructure
- TestAPS8IDISpecificPaths
- TestBayesianIntegration
- safe_version
- TestInterpolateG2Data
- test_tg6_fitting_p1.py
- get_health_monitor
- run_comprehensive_validation
- TestMemoryAndResourceErrors
- TestCommandPaletteExecution
- ._calculate_q_magnitude
- ScientificValidationFramework
- TestDragDropListViewInit
- test_recent_paths.py
- RecentPath
- TestToastStyling
- TestGetToolColor
- TestDefaultDrawParams
- TestWindowPartitionControls
- ensure_numpy
- QtErrorCapture
- fixture
- ScientificValidationTestSuite
- gui
- TestSimpleMaskUnsavedChanges
- 2.1 HDF5 File I/O
- 3.1 Core Data Structures
- ._save
- .__init__
- .request_cancel
- ReliabilityContext
- .get_module
- TestG2ModConstants
- TestErrorHandling
- MockH5pyFile
- test_analysis_tabs.py
- TestG2AnalysisTab
- TestUIBoundaryConditions
- TestEdgeCaseData
- TestFileListManagement
- TestDragAndDrop
- TestSimpleMaskDataLoading
- StatisticalCrossValidator
- TestViewerKernelInit
- test_aps_8idi.py
- TestAPS8IDICompatibility
- TestAPS8IDIKeyValidation
- TestAPS8IDIDataTypes
- TestBug022DoubleExpTauSorting
- TestDragDropListViewWithModel
- RecentPathsState
- 4. Compound Bugs — Issues That Must Be Fixed Together
- 6. Architecture Improvement Recommendations
- test_shortcut_manager.py
- comprehensive_xpcs_hdf5 (fixture)
- twotime_utils.py
- TestShortcutQuery
- TestShortcutConflictDetection
- .get_memory_usage
- test_drawing_tools.py
- TestToolColors
- TestQMapCacheNoCopy
- XPCS logo 128x128
- get_qmap
- TestHealthMonitorGCDelta
- TestNonzeroNoRecompilation
- ThreadSafeQtDecorator
- TestStabilizer
- scientific_validation.py
- ._populate_results
- .saxs_2d
- 10. Conclusion
- 1. Internal Module Dependencies
- 4. Cross-Module Data Flows
- 6. Recommended Architecture Patterns
- 7. Migration Roadmap
- 9. Performance Implications
- Appendix A: Data Structure Reference
- .setup_ui
- test_parametrized_invalid_data_types
- BUG-018: stretched_exp_model clamps tau
- TestMatplotlibIntegration
- test_file_operations.py
- TestCommandPaletteKeyboard
- .window_and_manager
- .__init__
- .start_operation
- TestDiagonalCorrectionPerformance
- TestDataGenerator
- TestDebugger
- TestSAXS1DTab
- TestTwoTimeTab
- TestStabilityTab
- qt_application (fixture)
- TestMetadataTab
- TestTabIntegration
- TestSignalSlotErrors
- TestViewerKernelProperties
- TestProgressIndication
- TestQtCompatWithPyQt6
- TestMaskExportContent
- .generate_validation_report
- TestPartitionSignalExport
- TestViewerKernelPerformance
- TestAsyncG2ResultHandling
- TestAPS8IDIKeyAccess
- BayesianDiagnosisWindow
- System Architecture Overview
- TestQMapUtilityMethods
- TestQMapEdgeCases
- TestXpcsFileAttributeCollision
- BUG-019: power_law_model LogNormal prior
- calibration.py
- test_drag_drop_list.py
- DragDropListView
- TestDragDropListViewSignals
- TestRecentPathsManagerGetPaths
- TestRecentPathsManagerRemoveInvalid
- TestNoScipyInterpolateImports
- TestG2Interpolation
- TestQmapOverlay
- ProgressIndicator
- TestIntegratedQtErrorScenarios
- .from_numpy
- improve_control_panel_layout
- TestQMapTab
- TestAverageTab
- verify_diffusion_constraints
- TestQMapCaching
- TestRecentPathsManagerClear
- Multi-Layered Test Framework
- .astype
- xpcsviewer/gui/__init__.py
- ._load
- twotime_batch.py
- .__init__
- .do_work
- .update_progress
- .stop_monitoring
- rerun_baselines.sh
- .single_exp_results
- create_separator
- .__init__
- BUG-G: tauq error bars use tauq_tau_err
- g2_analysis/__init__.py
- tests/analysis/__init__.py
- error_handling/__init__.py
- framework/__init__.py
- runners/__init__.py
- gui_interactive/__init__.py
- integration/gui/__init__.py
- autograd/__init__.py
- backend/__init__.py
- jax_migration/fitting/__init__.py
- jax_migration/__init__.py
- jax_migration/integration/__init__.py
- numerical/__init__.py
- performance/__init__.py
- precision/__init__.py
- visualization/__init__.py
- unit/analysis/__init__.py
- core/__init__.py
- fileio/__init__.py
- unit/fitting/__init__.py
- unit/gui/__init__.py
- .test_font_sizes_are_ascending
- .test_theme_has_all_components
- BUG-012: cancel_all_operations must disconnect signal-slot pairs before clearing
- BUG-031: WorkerManager.active_workers dict access must be protected with threading.Lock
- capture_logs (fixture)
- .test_theme_definitions_are_immutable
- .test_dark_colors_are_valid_hex
- .test_color_tokens_are_immutable
- .test_wcag_contrast_dark_text
- .test_spacing_follows_8px_grid
- .test_spacing_tokens_are_immutable
- XPCS Viewer Logo
- SAXS 1D Profile
- .test_spacing_values_are_integers
- AverageToolbox
- .test_spacing_values_are_positive
- Performance Baseline Profile
- .test_default_font_family
- unit/__init__.py
- unit/simplemask/__init__.py
- unit/threading/__init__.py
- upstream_features/__init__.py
- TestG2PlotBug002
- unit/utils/__init__.py
- tests/utils/__init__.py
- .arctan
- .arctan2
- .bincount
- .clip
- .concatenate
- .deg2rad
- .exp
- .fori_loop
- .logical_not
- activity.svg (activity/status indicator icon)
- .logspace
- DeviceManager
- BackendProtocol
- BUG-038 (_minimize_optax python loop not JIT-compiled)
- CIIntegration
- G2 interp cold: 354.5ms, warm: 0.43ms
- .mean
- performance_timer (fixture)
- temp_dir (fixture)
- generate_detailed_report
- generate_html_report
- generate_json_report
- generate_xml_report
- Contributing Guide
- Documentation Structure
- Installation Snippet (pip / uv)
- XPCS Viewer Usage Documentation
- SamplerConfig
- gui_test (decorator)
- performance_test (decorator)
- scientific_test (decorator)
- .nanmax
- .nanmin
- .percentile
- .rad2deg
- .round
- tab-setup icon
- Average Intensity Plot
- G2 Model Visualization
- Integrated Intensity Plot
- Offscreen GUI Snapshot
- Interp1d
- .scan
- .sqrt
- XPCS Viewer Logo (JPG)
- .stack
- .std
- .transpose
- test_g2mod.py
- configure_pyqtgraph_for_qt_compatibility
- .where
- Architecture Map Report
- Type Design Audit
- vectorized_background_subtraction
- test_g2_saxs_opt.py
- .abs
- .ceil
- .clip
- .concatenate
- TestBayesianAssembly
- TestExportBayesianCsv
- TestBugB_Jitter
- TestBugG_PowerLaw
- .copy
- TestGPULaunch
- TestHDF5ConnectionPool
- TestModelFunctionEquivalence
- test_nlsq_jit_tracing.py
- TestQtJaxInterop
- .deg2rad
- .digitize
- .exp
- .hypot
- twotime_batch module
- test_batch_vectorize.py
- TestC2StatisticsVectorized
- .isfinite
- .isnan
- .logical_not
- .logical_or
- .logspace
- XPCS Analysis Pipeline
- xpcs_logo_256x256.png (application logo 256x256)
- generate_synthetic_c2_matrix
- generate_synthetic_saxs_2d
- .meshgrid
- .nanmax
- .nanmean
- .nanmin
- .nonzero
- .rad2deg
- .reshape
- .std
- .sum
- .to_numpy
- .transpose
- module.saxs2d
- HDF5Adapter
- MatplotlibAdapter
- PyQtGraphAdapter
- BasePlotWorker
- BUG-017: getattr success default False
- BUG-024: FitResult __post_init__ validation
- BUG-025: FitDiagnostics __post_init__ validation
- BUG-007: Signal emissions from ThreadPoolExecutor threads must use invokeMethod
- BUG-009: AverageToolbox.is_killed must be threading.Event not plain bool
- BUG-020 PRNG non-deterministic seed
- BUG-021 per-chain init jitter
- CurveFitResult (native NLSQ 0.6.0)
- single_exp
- MATPLOTLIB_LIGHT
- Installation Guide
- refresh icon
- tab-correlation icon
- tab-scattering icon
- lumma_2000_g2_fitting (literature reference)
- ponmurugan_2009_sphere_form_factor (literature reference)
- ReferenceValidator
- SimpleMaskWindow
- TestG2Mod
- TestTwoTimeAnalysis
- cprofile_hotpaths script
- TestBatchVectorize
- TestBottleneck2MaskAssembleCopies
- TestBottleneck3BatchG2Normalization
- TestBottleneck5NLSQRobust
- TestModelsBaseline
- run_coverage_analysis
- jax_backend fixture
- numpy_backend fixture
- TestHDF5JaxIO
- SC-004: 1e-6 relative tolerance for Q-values
- SC-001: Q-map 5x faster on GPU vs CPU (2048x2048)
- T083: Angular computation precision test
- T084: Q-map tolerance test
- T093: ArviZ diagnostic plot generation test
- FR-020: plot_comparison overlaying NLSQ and Bayesian results
- T095: Comparison plot test
- T091: NLSQ uncertainty band computation test
- T092: pcov validation test
- FR-018: save_figure at 300 DPI
- T096: Plot export test
- T094: Posterior predictive plot test
- NLSQ single_exp cold: 1830.3ms, warm: 463.0ms
- Q-map 512x512 cold: 134.8ms, warm: 4.28ms
- Segmentation fault in test_main_window.py::test_g2_analysis_tab
- TestBugA_InitParams
- TestBugC_GetHdi
- TestBugE_TimeSeed
- TestBugF_Predict
- TestRecentPathsPersistence
- TestSessionPath
- TestThemeManagerInit
- TestThemeManagerSpacing
- TestTauqFallback
- get_qmap
- FitDiagnostics
- FitResult
- PUBLICATION_STYLE constant
- ThemeMode
- tab_mapping
- TestCloseEventWaitsForThreadPool
- TestIntensityTimeTab

## God Nodes (most connected - your core abstractions)
1. `XpcsViewer` - 211 edges
2. `XpcsFile` - 184 edges
3. `FitResult` - 122 edges
4. `ViewerKernel` - 117 edges
5. `NLSQResult` - 113 edges
6. `get_backend()` - 112 edges
7. `SimpleMaskKernel` - 98 edges
8. `JAXBackend` - 87 edges
9. `FileLocator` - 85 edges
10. `ensure_numpy()` - 84 edges

## Surprising Connections (you probably didn't know these)
- `Threading Module (Async Workers, Bayesian Worker)` --ui_action_open_files--> `folder-open icon`  [INFERRED]
  docs/api/threading.rst → xpcsviewer/ui/resources/icons/folder-open.svg
- `Threading Module (Async Workers, Bayesian Worker)` --ui_represents_mask_editor_tab--> `grid-mask icon`  [INFERRED]
  docs/api/threading.rst → xpcsviewer/ui/resources/icons/grid-mask.svg
- `MockH5py` --uses--> `XpcsFile`  [INFERRED]
  tests/error_handling/test_data_validation_errors.py → xpcsviewer/xpcs_file.py
- `MockH5py` --uses--> `XpcsFile`  [INFERRED]
  tests/error_handling/test_edge_cases.py → xpcsviewer/xpcs_file.py
- `MockH5py` --uses--> `XpcsFile`  [INFERRED]
  tests/gui_interactive/test_file_operations.py → xpcsviewer/xpcs_file.py

## Import Cycles
- 3-file cycle: `xpcsviewer/__init__.py -> xpcsviewer/xpcs_viewer.py -> xpcsviewer/viewer_ui.py -> xpcsviewer/__init__.py`

## Communities (723 total, 257 thin omitted)

### Community 0 - "plot_posterior_predictive"
Cohesion: 0.09
Nodes (25): mock_fit_result(), fixture, skipif, Tests for posterior predictive plot with 95% CI (T094). Tests for…, Test credible interval band is plotted., Test custom credible level works., Test custom n_draws parameter works., Test legend is present. (+17 more)

### Community 1 - "AsyncViewerKernel"
Cohesion: 0.02
Nodes (90): BUG-012: cancel_all_operations() disconnects all tracked signal connections., BUG-031: WorkerManager protects active_workers with a threading.Lock., BUG-031: Concurrent insertion and removal of active_workers is race-free., BUG-012 + BUG-031: Cancellation followed by re-submission does not leave stale…, BUG-008 + BUG-033: Concurrent HDF5 reads from multiple threads do not deadlock., Unit tests for threading signal safety fixes. Tests for BUG-007, BUG-012, and…, BUG-012: cancel_all_operations() must clean up _signal_connections., Create an AsyncViewerKernel with a mock thread pool. (+82 more)

### Community 3 - ".robust_curve_fit"
Cohesion: 0.05
Nodes (36): skipUnless, Backward compatibility tests for robust fitting framework. This module ensures…, Test compatibility with scipy.optimize.curve_fit interface., Set up scipy compatibility tests., Test that robust_curve_fit can replace scipy.optimize.curve_fit., Test that all scipy curve_fit parameters work with robust_curve_fit., Test that error handling is compatible with scipy expectations., Test compatibility with existing g2mod module. (+28 more)

### Community 4 - "NLSQResult"
Cohesion: 0.02
Nodes (65): ModelHealthReport, BUG-023: Failed NLSQResult metrics are NaN, distinguishable from bad-but-…, fixture, Tests for NLSQResult delegation to CurveFitResult (US1 - T008-T017). This…, Test predictions delegates to native_result and returns numpy array., Test confidence_intervals property delegates to native_result., Test get_confidence_interval method for single parameter., Test get_prediction_interval delegates to native_result. (+57 more)

### Community 5 - "nlsq_optimize"
Cohesion: 0.03
Nodes (90): MCMC, model_benchmark_data(), fixture, Baseline benchmarks for model functions. Establishes performance baselines…, Generate data for model function benchmarks., End-to-end workflow tests for JAX migration (SC-005). SC-005: 100% of existing…, End-to-end tests for Bayesian fitting workflow., Test complete Bayesian fitting from data to diagnostics. Workflow: 1. Generate… (+82 more)

### Community 7 - "ViewerKernel"
Cohesion: 0.05
Nodes (23): FileLocator, Test that plot_kwargs_record is properly initialized., Test memory cleanup threshold setting., Test suite for ViewerKernel metadata management., Test reset_meta creates correct metadata structure., Test that reset_meta maintains dictionary type., Test suite for ViewerKernel memory management features., Test that current_dset_cache is properly initialized. (+15 more)

### Community 8 - "XpcsViewer"
Cohesion: 0.03
Nodes (31): SimpleMaskWindow, take_snapshot, Ui, ensure_numpy, main_gui, setting, Main XPCS Viewer application window. XpcsViewer provides a comprehensive GUI…, Show the progress dialog. (+23 more)

### Community 9 - "FitResult"
Cohesion: 0.01
Nodes (172): _make_hdf5_g2_file(), Integration tests for compound bug chains. Verifies that all 6 cross-component…, BUG-025: FitDiagnostics __post_init__ rejects negative divergences., BUG-025: FitDiagnostics __post_init__ rejects negative ESS values., BUG-024: FitResult __post_init__ rejects empty samples dict., BUG-024: FitResult __post_init__ rejects inconsistent sample array shapes., Full chain: missing Q values -> dummy Q (correct axis) -> NLSQ sentinel…, Integration test for signal safety during shutdown. Verifies: -… (+164 more)

### Community 11 - "SessionManager"
Cohesion: 0.03
Nodes (87): fixture, Integration tests for session persistence functionality., Valid files should not generate warnings., Corrupted session file should not crash application., Empty session file should be handled gracefully., Session restore should complete within 3 seconds for 20 files., Integration tests for session save/restore cycle., Clearing session should remove session file. (+79 more)

### Community 12 - "xpcs_file.py"
Cohesion: 0.06
Nodes (47): get_enhanced_hdf5_reader(), get_enhanced_reader(), Get or create the global enhanced HDF5 reader., Alias for get_enhanced_hdf5_reader for backward compatibility., get_lazy_loader(), Lazy Loading System for XPCS Data This module provides smart data loading that…, Get or create the global lazy loader instance. Uses double-checked locking to…, Convenience function for registering HDF5 data for lazy loading. (+39 more)

### Community 14 - "._refresh_display"
Cohesion: 0.10
Nodes (13): float64, NDArray, Toggle mask overlay visibility., Toggle Q-map overlay visibility., Toggle partition overlay visibility., Update geometry spinboxes from metadata., Refresh display with optional overlays (mask, Q-map, or partition)., Create display image with Q-map overlay. Args: image: Base image (2D) qmap:… (+5 more)

### Community 15 - "BaseAsyncWorker"
Cohesion: 0.02
Nodes (121): integration, QRunnable, Unit tests for async workers module. This module provides comprehensive unit…, Test WorkerSignals has expected signal attributes., Test suite for BaseAsyncWorker class., Test BaseAsyncWorker can be instantiated., Test BaseAsyncWorker has expected methods., Test BaseAsyncWorker state management. (+113 more)

### Community 17 - "xpcs_viewer.py"
Cohesion: 0.02
Nodes (159): Structured Logging System, Threading API Reference, F, Formatter, Threading Audit Report, End-to-End Workflow Tests for XPCS Toolkit This package contains comprehensive…, Core test fixtures for XPCS Toolkit tests. This module provides basic fixtures…, Suppress common scientific computing warnings during tests. (+151 more)

### Community 20 - "tests/conftest.py"
Cohesion: 0.04
Nodes (79): auto_performance_monitoring(), corrupted_hdf5_file(), disk_space_exhausted_environment(), disk_space_limited_environment(), edge_case_data(), error_injector(), error_temp_dir(), file_handle_exhausted_environment() (+71 more)

### Community 21 - "QMapSchema"
Cohesion: 0.02
Nodes (98): Unit tests for XPCS schema validators. This module tests the schema validation…, Invalid phis unit should raise ValueError., Test QMapSchema shape validation., Mismatched sqmap/dqmap shapes should raise ValueError., Mismatched mask shape should raise ValueError., Test G2Data schema validation., Valid G2Data instantiation., Mismatched g2/g2_err shapes should raise ValueError. (+90 more)

### Community 23 - "JAXBackend"
Cohesion: 0.03
Nodes (68): _ensure_jax(), JAXBackend, Any, ndarray, JAX backend implementation with GPU and JIT support. This backend provides GPU-…, Create array with linearly spaced values., Create array with logarithmically spaced values., Create coordinate matrices from coordinate vectors. (+60 more)

### Community 25 - "XpcsFile"
Cohesion: 0.04
Nodes (29): Test XpcsFile initialization with invalid file., Test XpcsFile initialization with missing file., TestXpcsFileStrRepresentation, safe_shutdown, create_xpcs_dataset, create_id(), dtype, Estimate memory usage of a numpy array in MB. Parameters ---------- shape :… (+21 more)

### Community 26 - "run_tests"
Cohesion: 0.67
Nodes (3): generate_report, run_linting, run_tests

### Community 29 - "FileLocator"
Cohesion: 0.04
Nodes (43): Test FileLocator with invalid directory path., Test FileLocator with directory permission issues., Test file scanning behavior with corrupted files present., parametrize, Test basic FileLocator initialization., Test that initialization creates required components., Test suite for FileLocator path management., Test set_path method. (+35 more)

### Community 30 - "ROIParameters"
Cohesion: 0.07
Nodes (42): MemoryEfficientIterator, Memory-efficient iterator for large arrays with automatic cleanup., calculate_multiple_rois_parallel(), calculate_pie_roi(), calculate_ring_roi(), ParallelROIProcessor, PieROICalculator, ABC (+34 more)

### Community 31 - "SimpleMaskKernel"
Cohesion: 0.03
Nodes (57): _make_ring_image(), Unit tests for SimpleMaskKernel.find_beam_center (auto beam-center wiring)., Synthetic detector image with a bright ring centered at true_center., TestFindBeamCenter, Unit tests for SimpleMask save/load functionality. Tests mask persistence to…, save_mask should handle no mask gracefully., Tests for load_mask functionality., load_mask should load mask from HDF5 file. (+49 more)

### Community 32 - "process_c2_batch"
Cohesion: 0.22
Nodes (10): ProcessPoolExecutor, get_all_c2_from_hdf_enhanced(), get_optimal_worker_count(), get_process_pool(), process_c2_batch(), ProcessPoolExecutor (module-level), Optimized batch processing of C2 matrices with vectorized operations. Args:…, Get optimal number of worker processes based on system resources. (+2 more)

### Community 33 - "compute_transmission_qmap"
Cohesion: 0.03
Nodes (63): skipif, End-to-end tests for partition generation workflow., Test complete partition generation from Q-map to binned regions. Workflow: 1.…, End-to-end tests for Q-map computation workflow., Test complete Q-map computation from parameters to output. Workflow: 1.…, End-to-end tests for complete integrated analysis pipeline., Test complete XPCS analysis pipeline from Q-map to fitted parameters. This is…, Test workflow stability when switching backends. (+55 more)

### Community 35 - "state_validator.py"
Cohesion: 0.05
Nodes (45): StateValidationLevel, AtomicCounter, get_state_statistics(), get_state_validator(), LockFreeStateValidator, Any, Enum, Lock-Free State Consistency Validation for XPCS Viewer. This module provides… (+37 more)

### Community 36 - "_get_module"
Cohesion: 0.06
Nodes (23): Create temporary directory for test files., temp_dir(), skipUnless, Test that lazy loading provides performance benefits., Test that lazy loading doesn't cause memory leaks., Basic test for thread safety of lazy loading., Test that module cache persists across multiple calls., Test suite for lazy loading functionality. (+15 more)

### Community 38 - ".is_memory_pressure_high"
Cohesion: 0.24
Nodes (6): log_timing, Clean up cached data and release memory resources. Performs comprehensive…, Generate G2 2D map visualization. Creates a 2D representation of G2 correlation…, Generate two-time correlation plots for dynamic analysis. Creates two-time…, Update averaging worker values with memory management. Handles dynamic…, Check if memory pressure is above threshold (static method for backward…

### Community 39 - "single_exp_func"
Cohesion: 0.03
Nodes (63): fitting_benchmark_data(), fixture, skipif, Benchmark tests for JIT-accelerated fitting functions. Verifies performance…, Generate data for fitting benchmarks., Benchmark tests for NLSQ JIT performance., Verify NLSQ fitting runs without error and returns valid result., Record timing for NLSQ fitting with JIT. (+55 more)

### Community 40 - "set_backend"
Cohesion: 0.05
Nodes (55): Tests for partition combination memory efficiency., Verify partition combination produces correct results., Record timing for partition combination., TestCombinePartitions, create_test_mask(), create_test_phi_map(), create_test_qmap(), ndarray (+47 more)

### Community 41 - "UnifiedMemoryManager"
Cohesion: 0.05
Nodes (45): cache_array(), cache_computation(), CacheEntry, CacheType, get_array(), get_computation(), Any, Enum (+37 more)

### Community 43 - "Ui_mainWindow"
Cohesion: 0.08
Nodes (22): GraphicsLayoutWidget, ImageView, QListView, icons_rc (Qt resources), FileLocator, Placeholder XpcsFile class for documentation builds., Placeholder FileLocator class for documentation builds., Placeholder ViewerKernel class for documentation builds. (+14 more)

### Community 44 - "HDF5ConnectionPool"
Cohesion: 0.04
Nodes (48): parametrize, patch, Unit tests for HDF5 reader module. This module provides comprehensive unit…, Test suite for HDF5ConnectionPool initialization., Test HDF5ConnectionPool initialization with default parameters., Test HDF5ConnectionPool initialization with custom parameters., Test suite for HDF5ConnectionPool basic operations., Test getting new connection. (+40 more)

### Community 45 - "xpcsviewer/simplemask/__init__.py"
Cohesion: 0.05
Nodes (47): Unit tests for area_mask module. Tests mask classes and MaskAssemble…, Disabled thresholds should not affect mask., Tests for MaskParameter class., Single AND constraint should mask outside range., OR constraint should include additional regions., Tests for MaskArray class., evaluate should create mask from boolean array., Tests for MaskBase class. (+39 more)

### Community 47 - "QHBoxLayout"
Cohesion: 0.09
Nodes (14): NavigationToolbar2QT, QHBoxLayout, MplCanvasBar, MplCanvasBarH, MplCanvasBarV, NavigationToolbarSimple, QWidget, Apply theme colors to this matplotlib canvas. Parameters ---------- theme : str… (+6 more)

### Community 48 - ".create_dataset"
Cohesion: 0.04
Nodes (37): MockH5pyFile, Test handling of incompatible data types., Test handling of infinite and NaN values in data., Test handling of negative values where positive are expected., Test data range validation errors., Test handling of extremely large numerical values., Test handling of invalid correlation times., Test handling of mismatched array dimensions. (+29 more)

### Community 49 - "ci_integration.py"
Cohesion: 0.05
Nodes (48): configure_ci_environment(), configure_test_environment(), pytest_configure(), Configure the test environment settings., Configure CI-specific settings., Configure pytest with custom markers and settings., ArtifactManager, ci_cleanup() (+40 more)

### Community 50 - "CoverageManager"
Cohesion: 0.06
Nodes (32): cleanup_old_data, CoverageManager, CoverageMetrics, CoverageReport, CoverageTarget, export_coverage_trends, Any, Path (+24 more)

### Community 52 - "patch"
Cohesion: 0.04
Nodes (41): fixture, parametrize, patch, Unit tests for XpcsFile class. This module provides comprehensive unit tests…, Test minimal XpcsFile initialization., Test XpcsFile initialization with custom qmap manager., Test suite for MemoryMonitor class., Test XpcsFile initialization with extra fields. (+33 more)

### Community 53 - "RateLimitedLogger"
Cohesion: 0.05
Nodes (34): Test rate limiting behavior under high-frequency logging (T093)., Verify rate limiter suppresses messages beyond rate limit., Verify rate limiter allows messages again after time passes., TestRateLimitingUnderLoad, Tests for RateLimitedLogger class., First message is always logged., Messages exceeding rate limit are suppressed., Rate limit recovers as time passes. (+26 more)

### Community 54 - "benchmark"
Cohesion: 0.05
Nodes (37): benchmark, fixture, skipif, Benchmark tests for Q-map computation (T061). Benchmarks for measuring JIT…, Benchmarks for partition computation., Create 512x512 Q-map for benchmarking., Create 512x512 mask for benchmarking., Benchmark partition with 36 linear bins. (+29 more)

### Community 55 - "qt_fixtures.py"
Cohesion: 0.05
Nodes (35): configure_qt_for_testing(), gui_test_helper(), mock_qt_signal(), mock_qt_thread(), MockQApplication, MockQt, MockQtSignal, MockQtThread (+27 more)

### Community 56 - "XPCSBaseError"
Cohesion: 0.07
Nodes (36): chain_exception(), convert_exception(), exception_context, handle_exceptions(), Any, Exception, Path, XPCS Viewer Exception Hierarchy for Enhanced Error Handling and Reliability.… (+28 more)

### Community 58 - "theme/__init__.py"
Cohesion: 0.06
Nodes (40): Unit tests for theme design tokens., Font sizes should be in reasonable range (8-36pt)., TypographyTokens should not have weight_* fields (removed — Qt QSS does not…, Tests for ThemeDefinition dataclass., Light theme should have correct name., Dark theme should have correct name., Light and dark themes should share spacing tokens., Light and dark themes should share typography tokens. (+32 more)

### Community 59 - "TestPartitionBlemishExport"
Cohesion: 0.04
Nodes (32): fixture, Unit tests for Blemish Map Export (Feature 5). Tests the blemish attribute in…, Number of blemish pixels should match loaded file., Tests for blemish inclusion in partition export., Create a SimpleMaskKernel mock for testing., Create a prepared kernel with data for partition computation., compute_partition should include 'blemish' key., Partition blemish should be a numpy array. (+24 more)

### Community 60 - "TestDataSpec"
Cohesion: 0.07
Nodes (33): AdvancedTestDataFactory, create_minimal_test_data(), create_performance_test_data(), create_realistic_xpcs_dataset(), get_test_data_factory(), MockH5py, Any, Path (+25 more)

### Community 61 - "XPCS Viewer (xpcsviewer) Python Package"
Cohesion: 0.06
Nodes (47): XPCS Analysis Modules (G2, SAXS1D, SAXS2D, Twotime, Stability, I(t), TauQ, Average), NumPy/JAX Backend Abstraction Layer, xpcsviewer.backends.scipy_replacements, Chu et al. 2022 - pyXPCSviewer JSR Paper, Detector Geometry Parameters (bcx, bcy, distance, pixel size, energy), CLI API Reference, GUI Components API Reference, GUI Modernization (PySide6, themes, command palette) (+39 more)

### Community 62 - "plot_nlsq_fit"
Cohesion: 0.07
Nodes (29): fixture, Tests for NLSQ visualization enhancements (T070-T073). Tests programmatic…, Test that prediction interval is not shown by default., Tests for diagnostics display (T072)., Test that metrics are shown by default., Test that metrics annotation contains expected values., Tests for diagnostics 2x2 subplot layout (T073)., Test that plot_diagnostics creates a figure with 4 axes. (+21 more)

### Community 63 - ".validate_signal_connection"
Cohesion: 0.18
Nodes (10): ConnectionType, Any, QObject, Validate and establish a Qt5+ compliant signal/slot connection. Args: signal:…, Upgrade a Qt4-style connection to Qt5+ syntax. Args: obj: QObject containing…, Safely connect a signal to a slot with validation. Args: signal: Qt signal…, Safely disconnect a signal from a slot. Args: signal: Qt signal slot: Optional…, Scan an object for potential legacy signal connections. Args: obj: QObject to… (+2 more)

### Community 64 - "xpcsviewer/utils/reliability.py"
Cohesion: 0.06
Nodes (44): number, ValidationResult, Specific validation failures with detailed field information. Used for: - Input…, XPCSValidationError, clear_reliability_caches(), get_validation_cache(), Enum, ndarray (+36 more)

### Community 65 - "TestBottleneck1G2EnsembleStatistics"
Cohesion: 0.16
Nodes (12): flaky, g2_ensemble_statistics_baseline(), g2_ensemble_statistics_candidate(), Bottleneck #1: compute_g2_ensemble_statistics — O(Q*B^2*T) matmul + list copy., Candidate produces same ensemble_mean as baseline., Candidate temporal_correlation has same data, different container type., Baseline timing — current g2mod implementation., Candidate timing — proposed fix. (+4 more)

### Community 66 - "plot (tau-q)"
Cohesion: 0.08
Nodes (32): _make_hdl(), _make_xf(), ndarray, Regression tests for tauq.plot rendering. Tests cover: BUG-F: Fit line must…, ax.plot() must NOT be called when tauq_success is False., ax.plot() must NOT be called when tauq_fit_line is None., Verify error bars use tauq_tau_err when available., yerr in ax.errorbar() must match tauq_tau_err, not fit_val. (+24 more)

### Community 67 - "TestFileIOErrors"
Cohesion: 0.04
Nodes (25): MockH5pyFile, slow, Test pooled connection health checking with file deletion., Test batch reading with I/O errors., Test chunked dataset reading with errors., Test file info extraction with corrupted metadata., Test metadata reading with various error conditions., Test lazy loading error handling in XpcsFile. (+17 more)

### Community 68 - "TestCircleScaleHandles"
Cohesion: 0.04
Nodes (32): fixture, parametrize, scientific, Unit tests for Enhanced ROI Scale Handles (Feature 1). Tests the additional…, Rectangle should have updated default size [200, 150]., Tests for Ellipse ROI scale handles (8 handles: 4 midpoints + 4 corners)., Create a minimal SimpleMaskKernel mock for testing., Ellipse ROI should have 8 scale handles. (+24 more)

### Community 69 - "isolation.py"
Cohesion: 0.05
Nodes (33): isolated_test_environment(), isolation_manager(), MockH5py, MockH5pyFile, monitor_performance(), Any, fixture, Path (+25 more)

### Community 70 - ".get_selected_rows"
Cohesion: 0.06
Nodes (18): create_param_tree(), Return False and show guidance when no target data is available., Update only the G2 profile when Q-bin changes., Plot the G2 Map visualization., Plot G2 correlation functions with optional fitting. Generates multi-tau…, Refit G2 correlation functions with force_refit=True to bypass cache. This…, Plot G2 correlation functions with fitting overlay on the g2 fitting tab. This…, Export power law fitting results from diffusion analysis. (+10 more)

### Community 71 - "2.5 Threading Types (`xpcsviewer/threading/`)"
Cohesion: 0.04
Nodes (44): 1. Rating Table, 2.1 Schema Types (`xpcsviewer/schemas/validators.py`), 2.2 Fitting Types (`xpcsviewer/fitting/results.py`), 2.3 Mask Hierarchy (`xpcsviewer/simplemask/area_mask.py`), 2.4 Backend Types (`xpcsviewer/backends/`), 2.5 Threading Types (`xpcsviewer/threading/`), 2.6 XpcsFile (`xpcsviewer/xpcs_file.py`), 2. Detailed Analysis (+36 more)

### Community 72 - "ConnectionStats"
Cohesion: 0.06
Nodes (23): Test suite for ConnectionStats class., Test ConnectionStats initialization., Test recording connection creation., Test recording connection reuse., Test recording connection eviction., Test recording successful health check., Test recording failed health check., Test recording cache miss. (+15 more)

### Community 73 - "MemoryMonitor"
Cohesion: 0.07
Nodes (27): CacheItem, DataCache, Any, Data caching utilities for XpcsFile. This module provides LRU caching with…, Perform cleanup when system memory pressure is high., Store data in cache. Parameters ---------- file_path : str File path identifier…, Retrieve data from cache. Parameters ---------- file_path : str File path…, Clear all cached data. (+19 more)

### Community 75 - "BatchBayesianCoordinator"
Cohesion: 0.07
Nodes (27): mock_thread_pool(), fixture, Tests for BatchBayesianCoordinator., Progress signal should be emitted after each completion., single_q_error should be emitted on worker failure., A cancelled worker should decrement accounting like an error., If every worker is cancelled, all_finished should still fire., Create a mock QThreadPool that runs workers synchronously. (+19 more)

### Community 78 - "get_backend"
Cohesion: 0.04
Nodes (69): create_test_metadata(), skipif, Tests for Q-map numerical equivalence between NumPy and JAX backends. This…, Test that JAX produces identical reflection Q-map to NumPy., Test Q-map equivalence with various beam center positions., Test Q-map equivalence with various detector distances., Test Q-map equivalence with various X-ray energies., Test Q-map equivalence with large detector arrays. (+61 more)

### Community 79 - "Contract Audit — Phase 2: Type and Contract Verification"
Cohesion: 0.05
Nodes (41): 1.1 `nlsq_optimize()` → `NLSQResult` — Caller verification, 1.2 `run_single_exp_fit()` → `FitResult` — Contract chain, 1.3 `NLSQResult` delegation — None-safety when `native_result=None`, 1.4 `SamplerConfig.__post_init__()` — Validation completeness, 1.5 `FitDiagnostics.converged` property, 1. Fitting Pipeline Contracts, 2.1 Schema enforcement at actual I/O boundaries, 2.2 `QMapSchema.from_dict()` — Missing key handling (+33 more)

### Community 80 - "safe_json_write"
Cohesion: 0.08
Nodes (32): Integration tests for theme switching functionality., Integration tests for theme preferences persistence., Preferences should save and load correctly., Default preferences should be created if file doesn't exist., TestThemePreferencesIntegration, Path, Tests for atomic I/O utilities., Tests for safe_json_write. (+24 more)

### Community 81 - "test_xpcs_file_data_access.py"
Cohesion: 0.04
Nodes (30): parametrize, Unit tests for XpcsFile G2 data access methods. This module tests the core G2…, Test G2 stability data retrieval (XF-004)., XF-004: Multi-frame G2 evolution returns frame indices and values., Stability data returns multiple frames., Test edge cases for G2 data retrieval., Edge case: Q-range contains NaN values should be handled., Verify mock properly tracks calls for testing. (+22 more)

### Community 82 - "Integration Points Catalog"
Cohesion: 0.05
Nodes (43): 1.1 Primary Data Files (XPCS Data), 1.2 SimpleMask Mask Files, 1.3 SimpleMask Partition Files, 1.4 Two-Time Correlation Cache, 1. HDF5 I/O Integration Points, 2.1 PyQtGraph Plotting Boundary, 2.2 HDF5 Writing Boundary, 2.3 Matplotlib Plotting Boundary (+35 more)

### Community 83 - "XPCS Toolkit GUI Interactive Tests"
Cohesion: 0.05
Nodes (39): 1. Main Window Tests (`test_main_window.py`), 1. Test Isolation, 2. Analysis Tab Tests (`test_analysis_tabs.py`), 2. Asynchronous Operations, 3. Mock Data Usage, 3. Plot Interaction Tests (`test_plot_interactions.py`), 4. Error Simulation, 4. File Operation Tests (`test_file_operations.py`) (+31 more)

### Community 84 - "QtTestRunner"
Cohesion: 0.06
Nodes (24): main(), _QtErrorCapture, QtTestRunner, Qt Test Runner for Error Detection. Specialized test runner for Qt-related…, Run a single Qt test function with error capture. Args: test_func: Test…, Run a suite of Qt test functions. Args: test_functions: List of test functions…, Generate summary of Qt errors across all tests., Specialized test runner for Qt GUI components with error capture. (+16 more)

### Community 85 - "gui"
Cohesion: 0.06
Nodes (29): gui, slow, Tests for complete user interaction scenarios and workflows. This module…, Test stability monitoring and analysis workflow., Test suite for complete analysis workflow scenarios., Test suite for complex multi-tab navigation scenarios., Test exploring all tabs in sequence with state validation., Test random tab switching to verify stability. (+21 more)

### Community 86 - "LoggingContext"
Cohesion: 0.03
Nodes (64): PathLike, Test session context correlation across log entries (T092)., Verify all logs within a context share the same session_id., Verify nested contexts maintain session continuity., Test path sanitization in log output (T094)., Verify home mode replaces home directory with ~., Verify hash mode hashes the filename while preserving extension., Verify none mode preserves the full path. (+56 more)

### Community 87 - "BenchmarkTimer"
Cohesion: 0.08
Nodes (26): benchmark_timer(), BenchmarkResult, BenchmarkTimer, compare_benchmarks(), Any, fixture, ndarray, Benchmark test configuration and fixtures. Provides timing fixtures and… (+18 more)

### Community 88 - "create_xpcs_dataset"
Cohesion: 0.10
Nodes (19): patch, Test suite for create_xpcs_dataset function., Test successful creation of XPCS dataset., Test handling of KeyError during dataset creation., Test suite for FileLocator get_hdf_info method., Test successful HDF info retrieval., Test HDF info retrieval without filter string., Test handling of IOError during dataset creation. (+11 more)

### Community 89 - "slice"
Cohesion: 0.09
Nodes (25): EnhancedHDF5Reader, Any, log_timing, ndarray, slice, Intelligent read-ahead cache for HDF5 data based on access patterns., Record an access for pattern analysis., Predict next likely access patterns for read-ahead. Parameters ----------… (+17 more)

### Community 91 - ".create_dataset"
Cohesion: 0.06
Nodes (24): data_integrity, edge_cases, stress, system_dependent, MockH5pyFile, scientific, Test handling of arrays at maximum reasonable dimensions., Test handling of minimum positive floating point values. (+16 more)

### Community 92 - "test_twotime_qbin_memory.py"
Cohesion: 0.08
Nodes (24): SimpleNamespace, _make_geometry(), _make_tab_widget(), End-to-end regression test for target-file session persistence. Guards the bug…, Finding 2 (adversarial review): even when the data dir is opened via a RELATIVE…, test_relative_data_path_persists_as_absolute(), test_target_files_survive_restart_from_other_cwd(), _FakeCombo (+16 more)

### Community 93 - "gui"
Cohesion: 0.06
Nodes (28): gui, slow, Tests for main window functionality and tab management. This module tests the…, Test two-time correlation tab functionality., Test suite for window state persistence and management., Test that tab states are maintained during switching., Test window closing behavior., Test suite for main window functionality. (+20 more)

### Community 95 - "compute_uncertainty_band"
Cohesion: 0.08
Nodes (27): linear_model(), Tests for NLSQ uncertainty band computation (T091). Tests for…, Test uncertainty band for exponential model., Test band computation for single point., Tests for compute_prediction_interval function (NLSQ 0.6.0). Prediction…, Test output shapes match input., Test lower <= fit <= upper., Test prediction interval is wider than confidence interval. PI accounts for… (+19 more)

### Community 96 - "save_figure"
Cohesion: 0.06
Nodes (31): End-to-end tests for visualization workflow., Test visualization generation and export workflow., TestVisualizationWorkflow, fixture, skipif, Tests for plot export functionality (T096). Tests for save_figure() function…, Test saving in multiple formats in one call., Tests for publication style preset. (+23 more)

### Community 97 - "ThemeManager"
Cohesion: 0.07
Nodes (19): ThemeMode, QObject, Apply theme based on current mode setting., Detect the operating system theme preference. Returns: "light" or "dark" based…, Load and apply the combined stylesheet., Build combined stylesheet with token substitution. Returns: Combined and…, Load a QSS file from the styles directory., Substitute @token references in stylesheet. Tokens are referenced as… (+11 more)

### Community 98 - "test_viewer_kernel_export.py"
Cohesion: 0.05
Nodes (26): Unit tests for ViewerKernel export methods. This module tests the export…, Ring ROI addition returns ROI index., Test sector ROI addition (VK-009)., VK-009: Sector ROI geometry creation., Sector ROI with various angles., Test edge case: export to non-existent directory., Edge case: Export to non-existent directory should be handled., Export should be able to create missing directories. (+18 more)

### Community 100 - "Facade and Schema Infrastructure"
Cohesion: 0.06
Nodes (34): 1. Import Errors, 1. Schema Validators (`xpcsviewer/schemas/`), 2. HDF5 Facade (`xpcsviewer/io/hdf5_facade.py`), 2. Validation Errors, 3. Backend I/O Adapters (`xpcsviewer/backends/io_adapter.py`), 3. Circular Imports, Architecture Components, Available Adapters (+26 more)

### Community 101 - "plot_comparison"
Cohesion: 0.09
Nodes (26): Axes, mock_fit_result(), mock_nlsq_result(), fixture, skipif, Tests for comparison plot (T095). Tests for plot_comparison() function that…, Test function creates new axes if none provided., Test data points are plotted. (+18 more)

### Community 102 - "_cprofile_hotpaths.py"
Cohesion: 0.13
Nodes (14): main(), profile_func(), cProfile analysis script for the 6 hot paths. Outputs cumulative time breakdown…, Profile a function and print top 15 cumulative callers., Baseline benchmarks for two-time correlation cleaning (hot path #2)., Verify nan_to_num cleaning removes all NaN/inf., Benchmark C2 cleaning with nan_to_num (500x500)., Benchmark C2 cleaning with interpolation (500x500). (+6 more)

### Community 103 - "SyntheticXPCSGenerator"
Cohesion: 0.09
Nodes (28): create_detector_geometry(), create_qmap_data(), create_synthetic_g2_data(), create_synthetic_saxs_data(), Any, ndarray, Synthetic XPCS dataset generators for testing. This module provides…, Add realistic noise to correlation function. (+20 more)

### Community 104 - ".load_path"
Cohesion: 0.06
Nodes (17): Lazy import wrapper for sanitize_path., sanitize_path(), ndarray, Restore workspace state from saved session., Initialize async kernel when viewer kernel is ready., Update plot display based on current tab and selected files. Automatically…, Synchronous plot update (original behavior)., Clear plot display for specified tab when no files are selected. (+9 more)

### Community 105 - "XPCS Toolkit Error Handling & Edge Case Test Suite"
Cohesion: 0.06
Nodes (33): Adding New Error Scenarios, Advanced Test Options, Basic Test Execution, `conftest.py`, Core Error Handling Tests, Custom Error Injection, Edge Case and Boundary Testing, Error Handling Principles (+25 more)

### Community 106 - "test_viewer_kernel_plotting.py"
Cohesion: 0.05
Nodes (26): parametrize, Unit tests for ViewerKernel plotting methods. This module tests the plotting…, SAXS 1D plotting receives correct handler arguments., Test SAXS 2D plotting with ROI overlay (VK-004)., VK-004: SAXS 2D plotting with ROI overlay., SAXS 2D plotting works without ROI., Test TwoTime heatmap plotting (VK-005)., VK-005: TwoTime heatmap rendering with C2 matrix. (+18 more)

### Community 107 - "legacy.py"
Cohesion: 0.08
Nodes (45): The factory must return a callable suitable for curve_fit., Factory-produced function must evaluate single exponential correctly., double_exp(), double_exp_all(), _fit_single_qvalue(), fit_with_fixed(), fit_with_fixed_parallel(), fit_with_fixed_sequential() (+37 more)

### Community 108 - "ArrayType"
Cohesion: 0.06
Nodes (15): ArrayType, Element-wise ceiling., Compute mean, ignoring NaN values., Compute minimum along axis., Return indices of bins to which each value belongs., Find unique elements of array., Element-wise logical AND., Element-wise logical OR. (+7 more)

### Community 110 - "gui"
Cohesion: 0.07
Nodes (24): gui, slow, Tests for plot widget interactions and visualization components. This module…, Test dynamic Matplotlib plot updates., Test suite for plot customization and styling., Test plot axis label customization., Test plot grid on/off functionality., Test different plot color schemes. (+16 more)

### Community 111 - "logging_config.py (LoggingConfig, get_logger, initialize_logging)"
Cohesion: 0.08
Nodes (33): ColoredConsoleFormatter (ANSI-colored console log formatter), get_logger() (factory returning module logger), HealthMonitor (system resource health tracking with callbacks), IntelligentLazyLoader (access-pattern-aware HDF5 prefetcher), LazyHDF5Array (lazy proxy for HDF5 array datasets), log_timing() (decorator for method timing with optional threshold), LoggingConfig (dataclass for logging configuration), LoggingContext (context manager for correlated log entries) (+25 more)

### Community 112 - "RLock"
Cohesion: 0.29
Nodes (3): RLock, File, Get or create a lock for a specific file. Tracks last-access time per entry and…

### Community 113 - "test_xpcs_file_fitting.py"
Cohesion: 0.05
Nodes (26): Unit tests for XpcsFile G2 fitting methods. This module tests the G2 fitting…, Test Q-dependent tau fitting (XF-012)., XF-012: Q-dependent tau fitting returns power law parameters., Test edge case: flat/constant G2 data handling., Edge case: flat G2 data should be detected., Test edge case: NaN in Q-range., Edge case: Q-range with NaN should be detected., Test validation of G2 fitting results. (+18 more)

### Community 114 - "PooledConnection"
Cohesion: 0.08
Nodes (25): MockH5py, Comprehensive file I/O error handling tests. This module tests error conditions…, Test XpcsFile error handling., Test FileLocator error handling., Test behavior under resource exhaustion conditions., Test error recovery and resource cleanup., Test memory cleanup after allocation failures., Test that errors are properly logged and propagated. (+17 more)

### Community 115 - "style_helpers.py"
Cohesion: 0.11
Nodes (22): ButtonSize, ButtonStyle, apply_destructive_buttons(), apply_secondary_buttons(), QGroupBox, QPushButton, QWidget, Style helper utilities for XPCS-TOOLKIT GUI. This module provides functions to… (+14 more)

### Community 116 - "SimpleMaskWindow"
Cohesion: 0.07
Nodes (20): QMainWindow, Handle window close with unsaved changes prompt. Args: event: Close event, Apply all pending drawings to the mask., Handle undo/redo/reset mask action. Args: action: Action type, Handle save mask action with file dialog., Handle load mask action with file dialog and validation., Validate that mask file dimensions match current detector shape. Args:…, Enable mask overlay and sync all toggle states. Called after mask-modifying… (+12 more)

### Community 117 - "gui_interactive/conftest.py"
Cohesion: 0.08
Nodes (35): gui_accessibility_helper(), gui_error_simulator(), gui_interaction_recorder(), gui_main_window(), gui_parameter_tree(), gui_performance_monitor(), gui_plot_widget(), gui_state_validator() (+27 more)

### Community 118 - "Scientific Algorithm Validation Framework"
Cohesion: 0.06
Nodes (35): Adding New Algorithm Tests, Adding New Benchmarks, Adding New Properties, Algorithm Validation (`algorithms/`), Analytical Benchmarks, Continuous Integration, Cross-Validation, Cross-Validation (+27 more)

### Community 119 - "test_viewer_kernel.py"
Cohesion: 0.07
Nodes (26): parametrize, patch, Unit tests for ViewerKernel class. This module provides comprehensive unit…, Test suite for ViewerKernel inheritance from FileLocator., Test that ViewerKernel properly inherits from FileLocator., Test that ViewerKernel calls parent __init__., Test suite for ViewerKernel weak reference handling., Test that weak reference cache behaves correctly. (+18 more)

### Community 120 - ".__init__"
Cohesion: 0.06
Nodes (15): QKeySequence, Register a keyboard shortcut. Args: shortcut_id: Unique identifier for the…, Show startup dialog with recent directories and options., Set up keyboard shortcut to show progress dialog., Rebuild the two-time tab with a side-panel layout. Restructures from the…, Initialize the G2 Map tab with dynamically created widgets., Initialize the G2 Fitting tab with plot and fitting controls. This method: 1.…, Set up connections for async operations. (+7 more)

### Community 121 - "Threading and Reliability Audit Report"
Cohesion: 0.06
Nodes (31): Confirmed Correct Patterns, Debugger Agent Supplementary Findings (Phase 2), Executive Summary, Items Verified Correct by Debugger Analysis, P0-1: Missing `@Slot` decorators on `QMetaObject.invokeMethod` targets, P0-2: `cancel_all_operations` races with in-flight completion callbacks, P0 — Critical: Crash or data-loss risk, P1-1: `_emit_operation_progress` iterates `active_operations` without a lock (+23 more)

### Community 123 - "test_xpcs_file_roi.py"
Cohesion: 0.06
Nodes (23): Unit tests for XpcsFile ROI data extraction methods. This module tests the ROI…, Test ROI extraction with invalid parameters., Invalid ROI parameters should be handled gracefully., Empty ROI list should be handled., Test edge case: worker exhaustion in parallel ROI extraction., Edge case: max_workers limit exceeded should be handled., Test parallel extraction with many ROIs., Test single ROI data extraction (XF-007). (+15 more)

### Community 124 - "test_nlsq_bayesian_integration.py"
Cohesion: 0.08
Nodes (25): _make_synthetic_data(), ndarray, needs_numpyro, Integration tests: NLSQ and Bayesian power-law fitting consistency. Verifies:…, When p0 is None and b is fixed, use midpoint of bounds., Verify Bayesian path recovers known power-law parameters., run_power_law_fit must recover tau0 and alpha from synthetic data., run_power_law_fit must accept the bounds parameter. (+17 more)

### Community 125 - "BackendProtocol"
Cohesion: 0.06
Nodes (19): Protocol, BackendProtocol, Backend protocol interface for JAX/NumPy array operations. This module defines…, Element-wise sqrt(x^2 + y^2)., Compute sum along axis., Compute maximum along axis., Protocol defining the backend interface for array operations. Both NumPyBackend…, Return indices of non-zero elements. (+11 more)

### Community 126 - "backends/__init__.py"
Cohesion: 0.06
Nodes (23): Tests for environment variable configuration (T070). Tests environment variable…, Tests for memory-related environment variables., Test XPCS_GPU_MEMORY_FRACTION is respected., Test invalid memory fraction is handled gracefully., Tests for logging of environment variable configuration., Test backend selection is logged., Tests for environment variable configuration., Test XPCS_USE_JAX=1 enables JAX backend. (+15 more)

### Community 128 - "generate_arviz_diagnostics"
Cohesion: 0.08
Nodes (22): fixture, skipif, Tests for ArviZ diagnostic plot generation (T093). Tests for…, Tests for diagnostic plot content., Create simple trace for content tests., Test generated figures have axes., Tests for generate_arviz_diagnostics function., Create mock InferenceData for testing. (+14 more)

### Community 130 - "xpcsviewer.fitting"
Cohesion: 0.08
Nodes (26): ADR-004 Backend Abstraction, ADR-001 JAX Migration, ADR-002 NLSQ Migration, module.g2mod, module.twotime, ViewerKernel, XpcsFile (God Object), xpcsviewer.backends (+18 more)

### Community 131 - "QMapManager"
Cohesion: 0.08
Nodes (24): parametrize, Unit tests for QMap utilities module. This module provides comprehensive unit…, Test suite for QMapManager class., Test QMapManager initialization., Test getting new QMap., Test suite for QMap detector extent calculation., Test get_detector_extent method., Test suite for QMap compute_qmap method. (+16 more)

### Community 132 - "JSONFormatter"
Cohesion: 0.04
Nodes (39): Filter, Verify JSON fields have correct types for aggregation systems., Verify exception information is properly structured., Test JSON formatter produces valid schema for log aggregation (T077)., Verify JSON formatter output is parseable JSON., Test parsing multiple JSON log entries., Verify multiple JSON entries can be parsed line-by-line., Verify entries can be filtered by any structured field. (+31 more)

### Community 133 - "ndarray"
Cohesion: 0.07
Nodes (18): Any, ndarray, Combine this mask with another mask. Args: mask: Existing mask to combine with,…, Set the mask from a coordinate array. Args: zero_loc: Array of shape (2, N)…, Load mask from an HDF5 file. Args: fname: Path to HDF5 file key: Dataset key…, Create mask based on intensity thresholds. Uses backend abstraction for GPU…, Create mask based on Q-map constraints. Uses backend abstraction for GPU…, Set mask from an array. Args: arr: Boolean or integer array where nonzero =… (+10 more)

### Community 135 - "LazyHDF5Array"
Cohesion: 0.07
Nodes (18): IntelligentLazyLoader, LazyHDF5Array, Any, Convert to numpy array, loading if necessary., Lazy loading system with memory-pressure-aware cleanup., Register HDF5 dataset for lazy loading. Parameters ---------- data_key : str…, Get lazy data proxy by key., Called by a proxy on access; triggers cleanup under memory pressure. (+10 more)

### Community 136 - "._disconnect_signals"
Cohesion: 0.07
Nodes (18): Any, log_timing, Slot, Execute heavy computation asynchronously. Args: compute_func: Function to…, Cancel an active operation by ID., Cancel all active operations. Disconnects all tracked signal-slot pairs before…, Get the result of a completed operation., Handle data loading completion. (+10 more)

### Community 137 - "XPCS Viewer — Architecture Map"
Cohesion: 0.07
Nodes (27): 10. Backend Abstraction Layer, 11. Summary of Key Architectural Risks, 1. Module Dependency Graph, 2. Thread Boundary Diagram, 3. Data Flow: HDF5 -> XpcsFile -> ViewerKernel -> PlotWorkers -> PyQtGraph, 4. Signal/Slot Connection Map, 5. JIT Compilation Boundary Inventory, 6. Schema Validation Points (+19 more)

### Community 138 - "test_hdf5_facade.py"
Cohesion: 0.06
Nodes (21): Integration tests for HDF5Facade. This module tests the HDF5 facade…, Test HDF5Facade connection pool statistics., Connection pool statistics retrieval., Pool stats contain valid values., Test mock call tracking for facade methods., Verify read operations track calls., Verify write operations track calls., Test HDF5Facade error handling. (+13 more)

### Community 139 - "ToastWidget"
Cohesion: 0.08
Nodes (18): ToastWidget should have a label with the message., ToastWidget should have working opacity property., Tests for ToastWidget., ToastWidget should be created with message., ToastWidget should store the message., ToastWidget should store the toast type., ToastWidget should default to INFO type., ToastWidget should store duration. (+10 more)

### Community 140 - "validate_array_compatibility"
Cohesion: 0.08
Nodes (21): Tests for validation utility data integrity features. Tests for Technical…, Verify ValidationError on any length mismatch in multiple arrays., Test legacy validation function is deprecated., Verify legacy function emits DeprecationWarning., Test validate_array_compatibility raises ValidationError on mismatch (T019)., Verify ValidationError raised when arrays have different lengths., Verify error message includes array names when provided., Verify returns True when arrays have same length. (+13 more)

### Community 142 - "layout_helpers.py"
Cohesion: 0.11
Nodes (31): _add_labeled_spinbox(), apply_all_layout_improvements(), _apply_compact_density_to_sidebars(), improve_file_panel_layout(), improve_tab_content_spacing(), mark_primary_action_buttons(), optimize_g2map_tab(), QPushButton (+23 more)

### Community 143 - "Performance Optimization Summary Report"
Cohesion: 0.07
Nodes (26): 10. Reproduction, 1. Before/After: Wall Time (Warm Path), 2. Before/After: Cold Start (JIT Compilation), 3. Before/After: Peak Memory, 4.1 Q-map (`xpcsviewer/simplemask/qmap.py`), 4.2 Two-time C2 Cleaning (`xpcsviewer/module/twotime.py`), 4.3 G2 Vectorized Operations (`xpcsviewer/module/g2mod.py`), 4.4 SAXS Processing (`xpcsviewer/module/saxs1d.py`) (+18 more)

### Community 144 - "Decision"
Cohesion: 0.08
Nodes (24): ADR-001: JAX Migration and Backend Abstraction, Architecture, Consequences, Context, Decision, Environment Variables, Key Design Choices, Status (+16 more)

### Community 146 - "TestSAXSVectorizedOperations"
Cohesion: 0.08
Nodes (18): given, settings, Test sphere form factor against analytical solution, Test Guinier approximation for small q, Property-based test for form factor scaling, Test vectorized SAXS operations for correctness and performance, Create synthetic 1D SAXS intensity with realistic features, Helper method for sphere form factor (+10 more)

### Community 147 - "TestToastManager"
Cohesion: 0.06
Nodes (17): fixture, Tests for ToastManager., Create a main window for testing., ToastManager should be created with parent., ToastManager should start with no toasts., ToastManager should have 3000ms default duration., ToastManager should allow setting default duration., show_toast should create and display a toast. (+9 more)

### Community 148 - "MaskAssemble"
Cohesion: 0.09
Nodes (19): Tests for MaskAssemble class., MaskAssemble should initialize with workers., apply should add new mask state to history when mask changes., undo should move to previous mask state., undo should not go below mask_ptr_min., redo should move to next mask state., reset should clear history back to initial state., get_mask should return mask at current pointer. (+11 more)

### Community 149 - "TestG2PartialSafetyCheck"
Cohesion: 0.07
Nodes (18): fixture, Status bar should show message when g2_partial is unavailable., Status message should display for 3 seconds (3000ms)., Should not crash when statusbar is None., Should log info message when g2_partial is unavailable., Tests for g2_partial availability check before stability plotting., Tests for handling when no Multitau files are available., Create a mock statusbar. (+10 more)

### Community 150 - "QMap"
Cohesion: 0.09
Nodes (15): Test QMap initialization with exception., Test QMap initialization with default parameters., Test suite for QMap performance characteristics., Test QMap initialization performance., Test suite for QMap initialization., Test successful QMap initialization., TestQMapInit, TestQMapPerformance (+7 more)

### Community 151 - "jax_migration/conftest.py"
Cohesion: 0.10
Nodes (28): backend(), gpu_available(), jax_backend(), numpy_backend(), fixture, pytest_collection_modifyitems(), Pytest fixtures for JAX migration tests. Provides: - Backend fixtures…, Generate sample G2 correlation data for fitting tests. (+20 more)

### Community 152 - "test_qt_jax_interop.py"
Cohesion: 0.08
Nodes (19): skipif, Tests for Qt/JAX interoperability (T074). Tests that Qt widgets work correctly…, Test mask arrays convert properly for Qt display., Tests for Qt/JAX threading compatibility., Test backend can be accessed from main thread., Test multiple rapid computations don't cause issues., Tests for image data conversion for Qt display., Test image arrays convert correctly for display. (+11 more)

### Community 153 - "test_angular_computations.py"
Cohesion: 0.08
Nodes (19): skipif, Tests for angular computation precision (T083). Tests that angular computations…, Test Q increases with distance from center., Tests for sin/cos precision in angular computations., Test sin²x + cos²x = 1 precision., Test angle to Q conversion precision., Tests for hypot (Euclidean distance) precision., Test hypot produces correct Pythagorean results. (+11 more)

### Community 154 - "test_float32_vs_float64.py"
Cohesion: 0.08
Nodes (19): skipif, Tests for float32 vs float64 precision (T082). Tests that float64 precision is…, Test exponential functions maintain precision., Tests for Q-map computation precision., Test Q-map maintains float64 precision., Test Q-map precision for small angles near beam center., Tests for fitting precision., Test NLSQ residual computation precision. (+11 more)

### Community 155 - "get_icon"
Cohesion: 0.08
Nodes (19): QEvent, QIcon, QStyle, QTabBar, get_icon(), _load_svg_icon(), Path, Load an SVG, replacing ``currentColor`` with the active text color. (+11 more)

### Community 156 - "NumPyBackend"
Cohesion: 0.07
Nodes (15): NumPyBackend, NumPy backend implementation (CPU fallback). This backend provides the baseline…, Element-wise arctangent., NumPy-based backend for array operations. This backend provides CPU-only…, Compute maximum along axis., Find unique elements of array. Note: size parameter is ignored for NumPy (used…, Return elements chosen from x or y depending on condition., NumPy does not support GPU. (+7 more)

### Community 158 - "ndarray"
Cohesion: 0.07
Nodes (12): ndarray, Element-wise arctangent of y/x, handling quadrants., Round to given number of decimals., Compute mean along axis., Compute percentile along axis., Compute minimum along axis., Element-wise logical AND., Element-wise natural logarithm. (+4 more)

### Community 159 - "get_memory_manager"
Cohesion: 0.09
Nodes (18): AccessPattern, CacheEntry, IntelligentChunker, dtype, Enum, Enhanced HDF5 Reader with Intelligent Chunking and Read-ahead Caching This…, Check if accesses follow sequential pattern., Check if accesses follow block pattern. (+10 more)

### Community 160 - ".add_drawing"
Cohesion: 0.07
Nodes (16): Any, ndarray, ROI, Recompute Q-map from current metadata. Returns: Tuple of (qmap_dict, units_dict), Execute undo/redo/reset on mask history. Args: action: One of "undo", "redo",…, Apply a mask type and update current mask. Args: target: Mask type to apply, or…, Load mask from HDF5 file. Args: fname: Path to HDF5 file key: Dataset key for…, Display detector image with optional beam center marker. Args: cmap: Matplotlib… (+8 more)

### Community 161 - "JAX Backend Audit Report"
Cohesion: 0.08
Nodes (23): 1.1 Functions With JIT — Confirmed, 1.2 Functions Without JIT — Opportunities, 1.3 XLA Recompilation Risk — `static_argnums` Gaps, 1. JIT Compilation Audit, 2.1 vmap Usage — Confirmed, 2.2 Sequential Loops — vmap Opportunities, 2. vmap Audit, 3.1 Correct I/O Boundary Usage — Confirmed (+15 more)

### Community 162 - "2. P1 — High: Observable Wrong Behaviour / Reliability"
Cohesion: 0.08
Nodes (24): 2. P1 — High: Observable Wrong Behaviour / Reliability, P1-01 — Stale plot data applied to GUI after `cancel_all_operations`, P1-02 — `del self.active_plot_operations[operation_id]` raises `KeyError` after cancel, P1-03 — `WeakValueDictionary` read from worker thread without lock, P1-04 — `_g2_bayesian_worker_active` flag not reset on `signals.cancelled`, P1-05 — `HealthMonitor.stop_monitoring()` deadlocks: holds lock while joining thread, P1-06 — `TwotimePlotWorker` forks child processes while parent holds HDF5 connection, P1-07 — `SimpleMask` signal connections doubled on window re-creation (+16 more)

### Community 163 - "single_exp"
Cohesion: 0.09
Nodes (19): Tests for scipy vs NLSQ equivalence (T055-T058). Verifies that nlsq.curve_fit…, Tests for metrics equivalence (T058)., Test that R² is close to 1 for good fits., Test that RMSE is close to noise level., Single exponential model for testing (JAX-compatible)., Test that residuals are centered around zero., Tests for legacy function compatibility., Test that curve_fit can still return tuple for legacy code. (+11 more)

### Community 164 - "visualization.py"
Cohesion: 0.10
Nodes (16): Tests for pcov validation (T092). Tests for validate_pcov() function that…, Tests for validate_pcov function., Test validation passes for valid covariance matrix., Test validation fails for None covariance., Test validation fails for inf values., Test validation fails for nan values., Test validation fails for non-positive semi-definite matrix., Test validation passes for identity matrix. (+8 more)

### Community 165 - "TestGIXPCSPrecisionFormatting"
Cohesion: 0.10
Nodes (16): phi values should be formatted with 3 decimal places., alpha values should be formatted with 3 decimal places., x and y pixel values should be formatted with 3 decimal places., Implementation of get_qmap_at_pos for testing., Tests for boundary and edge cases in GIXPCS precision formatting., Position outside detector should return None., Very small qx values should still show 6 significant figures., Zero qx should be displayed as 0.000000. (+8 more)

### Community 166 - "ThreadingViolationDetector"
Cohesion: 0.09
Nodes (16): detect_threading_violations(), Restore original QTimer methods., Restore original QObject methods., Get current stack trace for violation context., Get all detected violations., Clear all recorded violations., Check if any timer violations were detected., Get summary of detected violations. (+8 more)

### Community 167 - "measure_memory"
Cohesion: 0.08
Nodes (17): measure_memory(), skipif, Measure peak memory allocation of a function call., Baseline benchmarks for G2 vectorized ops (hot path #3)., Benchmark vectorized baseline correction., Benchmark batch G2 normalization., Benchmark ensemble statistics computation (includes batched corrcoef)., Measure peak memory for ensemble statistics. (+9 more)

### Community 168 - "scientific_fixtures.py"
Cohesion: 0.11
Nodes (25): Test fixtures package for XPCS Toolkit. This package provides synthetic…, assert_arrays_close(), comprehensive_xpcs_hdf5(), correlation_function_validator(), create_test_dataset(), detector_geometry(), minimal_xpcs_hdf5(), Any (+17 more)

### Community 169 - "TestThemeSwitchingIntegration"
Cohesion: 0.07
Nodes (15): fixture, _build_stylesheet should return non-empty stylesheet., Light and dark stylesheets should be different., get_color should return valid color string., Colors should differ between light and dark themes., get_tokens should return current theme definition., Integration tests for theme switching across the application., Create a ThemeManager with temporary preferences. (+7 more)

### Community 170 - "Algorithmic Bottleneck Analysis"
Cohesion: 0.11
Nodes (22): Additional Findings, Algorithmic Bottleneck Analysis, `batch_g2_normalization` -- Python loop over stackable data, Bottleneck #1 — NLSQ multi-start: `preset="robust"` runs 5 TRF solves per fit call, Bottleneck #2 — `compute_g2_ensemble_statistics`: `np.median` = 85% of runtime, Bottleneck #3 — `clean_c2_for_visualization`: 3 separate sort passes = 67% of runtime, Complexity, Correction: SAXS q-binning is already fixed in production (+14 more)

### Community 171 - "test_user_defined_gradients.py"
Cohesion: 0.08
Nodes (17): skipif, Tests for user-defined gradient functions (T067a). Tests auto-diff with user-…, Tests for gradient API usability and ergonomics., Test backend.grad with multiple arguments., Test JIT compilation combined with gradient computation., Test gradients work through JAX control flow primitives., Tests for gradient edge cases., Test gradient handling of NaN values. (+9 more)

### Community 172 - "TestBackendDetection"
Cohesion: 0.07
Nodes (17): jax, Test explicit backend setting., Test setting NumPy backend explicitly., Test setting JAX backend explicitly., Test setting invalid backend raises ValueError., Test backend name is case-insensitive., Test backend detection and initialization., Test that get_backend returns a BackendProtocol instance. (+9 more)

### Community 173 - "test_hdf5_jax_io.py"
Cohesion: 0.08
Nodes (18): skipif, Tests for HDF5/JAX I/O (T075). Tests that HDF5 I/O works correctly with JAX…, Tests for writing results to HDF5 from JAX backend., Test writing Q-map results to HDF5., Test writing partition results to HDF5., Tests for mask handling with HDF5 and JAX., Test reading boolean mask from HDF5., Test writing mask to HDF5 after JAX processing. (+10 more)

### Community 174 - "TestPlotConstants"
Cohesion: 0.07
Nodes (15): skipUnless, Test the get_color_cycle function., Test the get_marker_cycle function., Test that the module provides backwards compatibility for common use cases., Test that accessing constants is performant (no heavy computation)., Test that constants work well for scientific plotting scenarios., Test that constants don't use excessive memory., Test suite for centralized plot constants. (+7 more)

### Community 175 - "h5py_mocks.py"
Cohesion: 0.09
Nodes (15): create_mock_hdf5_file(), create_mock_xpcs_file(), MockH5pyFile, MockXpcsFile, Path, Shared H5py mock objects for testing. This module provides standardized mock…, Set up default XPCS file structure., Mock flush operation. (+7 more)

### Community 176 - "TestC2StatisticsBaseline"
Cohesion: 0.10
Nodes (19): c2_statistics_data(), diagonal_correction_loop(), off_diagonal_stats_loop(), off_diagonal_stats_vectorized(), fixture, ndarray, Baseline benchmarks for C2 two-time correlation statistics. Establishes…, Verify loop and vectorized produce same results. (+11 more)

### Community 177 - "AnalyticalBenchmarkSuite"
Cohesion: 0.11
Nodes (14): AnalyticalBenchmarkSuite, ndarray, Single exponential G2 correlation function G2(τ) = baseline + β * exp(-γτ)…, Double exponential G2 correlation function G2(τ) = baseline + β₁*exp(-γ₁τ) +…, Suite of analytical benchmarks for XPCS algorithm validation, Stretched exponential (Kohlrausch-Williams-Watts) G2 correlation G2(τ) =…, Sphere form factor (Rayleigh scattering) F(q) = 3[sin(qR) - qR*cos(qR)]/(qR)³…, Infinite cylinder form factor (averaged over orientations) I(q) = I₀ *… (+6 more)

### Community 178 - "TestInitAverageSaveNamePreservation"
Cohesion: 0.25
Nodes (5): BUG-016: init_average must only set the default save_name when the text field…, init_average must check if save_name field is empty before overwriting., Simulate init_average behavior: non-empty name must be preserved., Simulate init_average behavior: empty name gets default., TestInitAverageSaveNamePreservation

### Community 180 - "test_calibration_baseline.py"
Cohesion: 0.11
Nodes (20): calibration_data(), compute_q_jax(), compute_q_numpy(), objective_jax(), objective_numpy(), fixture, ndarray, skipif (+12 more)

### Community 182 - "test_memory_limits.py"
Cohesion: 0.09
Nodes (16): skipif, Tests for memory limits during large computations (T073a). Tests that memory…, Test fitting doesn't exhaust memory., Tests for SC-007: Memory usage stays within 90% of available device memory., Test memory usage stays below 90% threshold per SC-007. SC-007: Memory usage…, Test GPU memory fraction is configured correctly., Test that large arrays are processed in chunks to manage memory., Tests for memory usage limits. (+8 more)

### Community 183 - "mathematical_invariants.py"
Cohesion: 0.11
Nodes (24): generate_valid_g2_data(), generate_valid_intensity_data(), ndarray, Mathematical Invariants for XPCS Analysis This module defines mathematical…, Verify G2 asymptotic behavior: G2(τ→∞) → baseline Args: g2_data: G2 correlation…, Verify G2 causality principle through time-reversal symmetry Args: g2_matrix:…, Verify that scattering intensities are non-negative Args: intensity: Intensity…, Verify form factor mathematical properties Args: q_values: Scattering vector… (+16 more)

### Community 184 - "TestQmapColormapUIWidget"
Cohesion: 0.11
Nodes (13): scientific, Unit tests for Qmap Colormap Selector (Feature 2). Tests the colormap dropdown…, Tests for the cb_qmap_cmap QComboBox widget in viewer_ui.py., Verify cb_qmap_cmap widget is defined in UI., Scientific tests for colormap visual properties., tab20b colors should be perceptually distinct for ROI visualization., viridis should provide smooth perceptual progression., Gray colormap should be strictly monotonic in luminance. (+5 more)

### Community 185 - "TestG2MathematicalProperties"
Cohesion: 0.09
Nodes (15): given, settings, Test that G2 respects causality, Test mathematical properties that G2 functions must satisfy, Test G2 fitting algorithms for accuracy and robustness, Set up fitting test data, Single exponential G2 model, Test single exponential fitting accuracy (+7 more)

### Community 186 - "TestCPUGPUNumericalEquivalence"
Cohesion: 0.11
Nodes (16): gpu, jax, Tests for numerical equivalence between CPU and GPU (US1). Tests FR-008:…, Test meshgrid produces equivalent results on CPU and GPU., Test Q-map-like computations for CPU/GPU equivalence., Test Q-map computation produces equivalent results., Test numerical equivalence between CPU and GPU computations., Test equivalence through the backend abstraction layer. (+8 more)

### Community 187 - "patch"
Cohesion: 0.11
Nodes (15): patch, Test suite for QMap load_dataset method., Test successful dataset loading., Test loading dataset when qmap group is missing., Test loading dataset with missing keys., Test suite for get_hash function., Test basic hash generation., Test that hash is consistent for same file properties. (+7 more)

### Community 188 - "TestQMapConstants"
Cohesion: 0.08
Nodes (13): skipUnless, Test common usage patterns with the constants., Test that constants don't use excessive memory., Test that accessing constants is performant., Test suite for centralized QMap constants., Test that the required constants exist., Test properties of the default detector size., Test properties of the default beam center. (+5 more)

### Community 189 - "XPCS Toolkit Test Suite"
Cohesion: 0.11
Nodes (19): Architecture Overview, Continuous Integration, Contributing, 🛠️ Developer Experience, Documentation, Key Features, Maintenance, 🔄 Maintenance & Evolution (+11 more)

### Community 190 - "TestBottleneck5NLSQMultiStart"
Cohesion: 0.12
Nodes (15): fit_multi_start(), fit_single_start(), Single TRF start from a reasonable initial guess., Multi-start TRF (simulates nlsq preset='robust')., Bottleneck #5 (baseline P0): nlsq_optimize multi-start overhead. Demonstrates…, Single start from ground truth region converges correctly., Multi-start converges correctly (ground truth as oracle)., Single-start and multi-start agree when single-start has good p0. (+7 more)

### Community 191 - "ndarray"
Cohesion: 0.15
Nodes (10): MaskHistoryBaseline, MaskHistoryCandidate, ndarray, Mimics MaskAssemble undo/redo pattern (current code)., Proposed pattern: remove the unnecessary copy in get_mask() for internal use.…, Internal: return read-only view (no copy)., Public API: always returns a writable copy (safe for external callers)., Bottleneck #2: MaskAssemble — excessive array copies in apply() / get_mask(). (+2 more)

### Community 193 - "run_gui_tests.py"
Cohesion: 0.15
Nodes (22): main(), Run user interaction scenario tests., Run CI-friendly GUI tests (headless, fast)., Run GUI tests with coverage reporting., Run a specific test file., Main test runner function., Set up environment variables for GUI testing., Run pytest with specified arguments. (+14 more)

### Community 194 - "TestBeamCenterCalibration"
Cohesion: 0.09
Nodes (15): skipif, Tests for gradient-based calibration (T063). Tests that gradient-based…, Test that detector distance gradient points in correct direction. This tests…, Tests for calibration convergence properties., Test that optimization converges at expected rate., Test that optimization can escape saddle points., Tests for integration with Optimistix optimizer., Test that optimistix is available for advanced optimization. (+7 more)

### Community 196 - "assemble_fit_summary"
Cohesion: 0.16
Nodes (13): _identity_model(), Test assemble_fit_summary end-to-end., Verify tau-q pipeline can read tau values via fit_val[:, 0, 1]., failed_mask must be in summary with correct shape and values., Empty results dict should produce all-NaN arrays., Simplified single-exp model for testing., TestAssembleFitSummary, assemble_fit_summary() (+5 more)

### Community 197 - "Baseline Performance Profile Report"
Cohesion: 0.20
Nodes (18): 1. Q-map Computation (`xpcsviewer/simplemask/qmap.py`), 2. Two-time Correlation C2 Cleaning (`xpcsviewer/module/twotime.py`), 3. G2 Vectorized Operations (`xpcsviewer/module/g2mod.py`), 4. SAXS 1D Processing (`xpcsviewer/module/saxs1d.py`), 5. NLSQ Fitting (`xpcsviewer/fitting/nlsq.py`), 6. HDF5 I/O / FFT Cache (`xpcsviewer/xpcs_file.py`), Analysis, Baseline Performance Profile Report (+10 more)

### Community 198 - "Numerical and JAX Audit Report"
Cohesion: 0.11
Nodes (18): Executive Summary, Non-Issues (Investigated but Dismissed), Numerical and JAX Audit Report, P0-001: Python control flow inside `@jax.jit` in reflection Q-map, P0 — Critical (Silent Correctness Failure), P1-001: `converged` always `False` due to missing `success` attribute, P1-002: NaN gradient from `jnp.clip(tau, 1e-30)` inside NumPyro model, P1-003: `model_fn(x, *popt)` unpacking JAX arrays inside NLSQ residual computation (+10 more)

### Community 199 - "MplCanvas"
Cohesion: 0.14
Nodes (4): adjust_yerr(), LineBuilder, MplCanvas, code copied from http://chuanshuoge2.blogspot.com/2019/12/matplotlib-mouse-…

### Community 200 - "CleanupScheduler"
Cohesion: 0.10
Nodes (15): CleanupScheduler, OptimizedCleanupSystem, log_timing, Thread, Schedules and manages cleanup operations., Schedule a cleanup task., Execute all pending cleanup tasks. Tasks are collected under the lock but…, Smart garbage collection with memory pressure awareness. (+7 more)

### Community 201 - "SmartFallbackManager"
Cohesion: 0.09
Nodes (16): get_fallback_manager(), get_reliability_profiler(), Any, Validate type with support for generic aliases. Args: value: The value to…, Manager for smart fallback strategies with pre-computed paths., Register a chain of fallback strategies for an operation., Get performance statistics for an operation., Get the global fallback manager instance. (+8 more)

### Community 202 - "TestTwoTimeCorrelationProperties"
Cohesion: 0.12
Nodes (13): given, settings, Test that correlation matrix is positive semi-definite, Test that diagonal elements represent one-time correlations, Test time translation invariance for stationary processes, Test relationship between time-averaged and ensemble-averaged correlations, Property-based test for correlation time estimation, Test mathematical properties of two-time correlation matrices (+5 more)

### Community 203 - "TestSaxsBinningBaseline"
Cohesion: 0.11
Nodes (16): loop_based_binning(), fixture, ndarray, Baseline benchmarks for SAXS 1D binning. Establishes performance baselines…, Verify loop and vectorized produce same results., Record baseline timing for loop-based binning., Record timing for vectorized binning (comparison)., Generate data for SAXS binning benchmarks. (+8 more)

### Community 204 - "batch_read_fields"
Cohesion: 0.11
Nodes (17): Test suite for get() function., Test getting simple field from HDF5 file., Test getting field with slice., TestGetFunction, batch_read_fields(), get(), get_chunked_dataset(), Any (+9 more)

### Community 205 - "TestSingletonDoubleCheckedLocking"
Cohesion: 0.10
Nodes (15): BUG-030: The three global singletons must use double-checked locking to be…, Helper: call getter_fn from many threads simultaneously and verify only one…, UnifiedThreadingManager singleton must be thread-safe under concurrency., MemoryManager singleton must be thread-safe under concurrency., LazyLoader singleton must be thread-safe under concurrency., get_unified_threading_manager source must contain a Lock., get_memory_manager source must contain a Lock., get_lazy_loader source must contain a Lock. (+7 more)

### Community 206 - "HDF5ValidationError"
Cohesion: 0.12
Nodes (15): _decode_h5_unit(), HDF5ValidationError, Any, Exception, log_timing, Path, Write mask to HDF5 file with versioning. Parameters ---------- file_path : str…, Write partition to HDF5 file with versioning. Parameters ---------- file_path :… (+7 more)

### Community 208 - "gui"
Cohesion: 0.13
Nodes (12): gui, Test handling of permission-denied file access., Test handling of files with malformed XPCS data structure., Test suite for calculation and analysis errors., Test handling of G2 fitting convergence failures., Test suite for data loading error scenarios., Test handling of numerical overflow in calculations., Test handling of division by zero in calculations. (+4 more)

### Community 209 - "TestQmapGradients"
Cohesion: 0.10
Nodes (13): skipif, Tests for Q-map gradient computation (T062). Tests that gradients can be…, Test backend.grad wrapper for gradient computation., Test backend.value_and_grad wrapper., Tests for gradient numerical accuracy., Test JAX gradients match finite difference approximation., Test second-order gradients (Hessian)., Tests for Q-map gradient computation. (+5 more)

### Community 210 - "constants.py"
Cohesion: 0.12
Nodes (11): TestG2AnalysisValidation, TestSAXSAnalysis, Two-Time Correlation Analysis Validation Tests This module provides…, # NOTE: This test primarily validates algorithm robustness rather than accuracy, Test physical constraints and interpretations of two-time correlations, Test that two-time correlations respect causality, Test relation between intensity fluctuations and correlations, Test properties specific to stationary processes (+3 more)

### Community 211 - "physical_constraints.py"
Cohesion: 0.11
Nodes (19): Mathematical Properties and Invariants Module This module defines mathematical…, ndarray, Physical Constraints and Laws This module defines physical constraints and laws…, Verify X-ray scattering physics constraints Args: q_values: Scattering vector…, Verify physics of intensity correlation functions Args: tau_values: Time delay…, Verify Stokes-Einstein relation: D = kT/(6πηr) Args: diffusion_coefficient:…, Verify conservation laws in scattering processes Args: input_intensity: Input…, Verify fluctuation-dissipation theorem for correlation and response functions… (+11 more)

### Community 212 - "ToastType"
Cohesion: 0.11
Nodes (13): Unit tests for ToastNotification widgets., Tests for ToastType enum., INFO type should have 'info' value., SUCCESS type should have 'success' value., WARNING type should have 'warning' value., ERROR type should have 'error' value., All expected toast types should exist., TestToastType (+5 more)

### Community 214 - "Any"
Cohesion: 0.10
Nodes (11): Any, Stack arrays along new axis., Scan over leading array dimension while carrying along state. Sequential…, Execute body function in a loop from lower to upper. Sequential fallback for…, Create array filled with zeros., Create array filled with ones., Create array with evenly spaced values., Create zero-filled array with same shape as input. (+3 more)

### Community 215 - "setter"
Cohesion: 0.10
Nodes (14): setter, Coefficient of determination (R²)., Set r_squared (for backward compat initialization). Note: This setter is a no-…, Adjusted R² accounting for number of parameters., Set adj_r_squared (for backward compat initialization). Note: This setter is a…, Root mean squared error., Set rmse (for backward compat initialization). Note: This setter is a no-op…, Set mae (for backward compat initialization). Note: This setter is a no-op when… (+6 more)

### Community 216 - "test_saxs_analysis.py"
Cohesion: 0.12
Nodes (13): SAXS Analysis Algorithm Validation Tests This module provides comprehensive…, Test batch processing of multiple SAXS datasets, Test Region of Interest (ROI) extraction from SAXS images, Set up test images and ROI definitions, Create synthetic SAXS image with known features, Test rectangular ROI extraction, Test circular ROI extraction, Test ROI extraction edge cases and error handling (+5 more)

### Community 217 - "reference_data/__init__.py"
Cohesion: 0.16
Nodes (19): create_analytical_diffusion_data(), create_analytical_g2_data(), create_analytical_saxs_data(), initialize_reference_data(), list_available_datasets(), load_reference_data(), Any, ndarray (+11 more)

### Community 218 - "test_simplemask_window.py"
Cohesion: 0.09
Nodes (14): GUI tests for SimpleMask window integration. Tests the SimpleMask window…, Tests for SimpleMask signal emission., mask_exported signal should exist., qmap_exported signal should exist., Tests for SimpleMask window creation and initialization., SimpleMask window should be created successfully., Tests for SimpleMask window close behavior., Window should close without prompt when no unsaved changes. (+6 more)

### Community 219 - "test_simplemask_integration.py"
Cohesion: 0.10
Nodes (13): Integration tests for SimpleMask to XPCS Viewer data flow. Tests the mask and…, Tests for mask_exported signal functionality., Tests for Apply to Viewer button in toolbar., Toolbar should have Apply to Viewer action., mask_exported signal should emit numpy array., File menu should have Apply to Viewer action., Tests for status bar messages during export., Exporting mask should update status bar. (+5 more)

### Community 220 - "test_backend_detection.py"
Cohesion: 0.12
Nodes (12): Tests for backend detection and initialization. Tests FR-001: Automatic device…, Test mathematical operations., Test environment variable configuration (FR-010)., Test XPCS_USE_JAX=false forces NumPy backend., Test XPCS_USE_JAX=true forces JAX backend., Test XPCS_USE_JAX=auto selects JAX when available., TestBackendMathOperations, TestEnvironmentConfiguration (+4 more)

### Community 221 - "TestDrawingToolsDefinitions"
Cohesion: 0.10
Nodes (11): Line tool should be defined., Ellipse tool should be defined., All drawing tools should have keyboard shortcuts., All drawing tools should have tooltips., All drawing tools should default to exclusive mode., Tests for DRAWING_TOOLS dictionary., DRAWING_TOOLS should contain tool definitions., Rectangle tool should be defined. (+3 more)

### Community 222 - "test_qmap_integration.py"
Cohesion: 0.10
Nodes (13): Unit tests for Q-map integration in SimpleMask window. Tests the Q-map…, Tests for Generate Q-Map toolbar action., Window should have Generate Q-Map toolbar action., Window should have Show Q-Map toggle action in toolbar., Tests for qmap_exported signal., qmap_exported signal should exist., export_partition_to_viewer method should exist., Tests for geometry spinboxes in SimpleMask window. (+5 more)

### Community 223 - "TestPlotQmapColormapIntegration"
Cohesion: 0.09
Nodes (15): fixture, parametrize, Tests for colormap application in ViewerKernel.plot_qmap., Create a mock ViewerKernel with required methods., Create a mock plot handler., plot_qmap should accept cmap parameter., plot_qmap should apply the specified colormap., plot_qmap should default to tab20b colormap. (+7 more)

### Community 224 - ".create_xpcs_file"
Cohesion: 0.12
Nodes (9): get_hdf5_manager(), HDF5TestDataManager, MockH5pyFile, File, Manager for creating and managing HDF5 test files., Create comprehensive XPCS HDF5 file., Create basic NeXus structure., Clean up temporary files. (+1 more)

### Community 225 - "Bayesian Fitting with NumPyro NUTS"
Cohesion: 0.14
Nodes (15): NLSQ 0.6.0 Migration (ADR-002), Bayesian Fitting with NumPyro NUTS, G2 Correlation Function, NLSQ Native CurveFitResult API, NumPyro NUTS Sampling, Two-Time Correlation Analysis, XPCS Technique, Diffusion Coefficient Plot (+7 more)

### Community 226 - "ImageViewDev"
Cohesion: 0.14
Nodes (5): ImageViewDev, PieROI, r""" Equilateral triangle ROI subclass with one scale handle and one rotation…, Apply theme colors to this image view. Parameters ---------- theme : str Either…, reset the viewbox's limits so updating image won't break the layout;

### Community 227 - "test_bottleneck_analysis.py"
Cohesion: 0.18
Nodes (11): batch_g2_normalization_baseline(), batch_g2_normalization_candidate(), make_g2_batch(), Micro-benchmarks isolating the top 3 algorithmic bottlenecks. Each section…, Return mean wall-clock time (ms) for fn() over n iterations., Verbatim copy of current implementation (g2mod.py:832-868)., Proposed fix: stack → single vectorised kernel → unstack. Complexity: O(B*T*Q)…, Bottleneck #3: batch_g2_normalization — Python loop over stackable data. (+3 more)

### Community 228 - "TestQmapBaseline"
Cohesion: 0.11
Nodes (12): measure_cold_warm(), slow, Measure cold (first call) and warm (subsequent calls) wall times., Baseline benchmarks for Q-map computation (hot path #1)., Verify transmission Q-map produces valid output., Benchmark transmission Q-map (512x512)., Measure cold vs warm timing for Q-map (includes JIT compilation)., Measure peak memory for Q-map computation. (+4 more)

### Community 230 - "TestDoubleExpBaseline"
Cohesion: 0.11
Nodes (13): skipif, Record baseline timing for double_exp_func., Baseline benchmark for stretched exponential function., Verify stretched_exp_func produces correct results., Record baseline timing for stretched_exp_func., Baseline benchmark for single exponential function., Verify single_exp_func produces correct results., Record baseline timing for single_exp_func. (+5 more)

### Community 232 - "statistical_properties.py"
Cohesion: 0.14
Nodes (18): ndarray, Statistical Properties and Constraints This module defines statistical…, Verify that fitting residuals are normally distributed Args: residuals: Fitting…, Verify that parameter uncertainties scale properly with sample size For most…, Verify bootstrap parameter estimation consistency Args: original_estimate:…, Verify parameter correlation structure in fitting Args: correlation_matrix:…, Verify Monte Carlo estimation convergence Args: estimates: Cumulative estimates…, Verify statistical power of hypothesis tests Args: test_statistics: Test… (+10 more)

### Community 233 - "3. P2 — Medium: Technical Debt / Maintainability"
Cohesion: 0.13
Nodes (15): 3. P2 — Medium: Technical Debt / Maintainability, P2-01 — `@abstractmethod` on `BackendProtocol` has no enforcement effect, P2-02 — `G2Data.to_dict()` returns raw (mutable) array references, P2-03 — Silent `"deg"` default for missing `phis_unit` in `QMapSchema`, P2-04 — `NLSQResult` lacks invariant enforcement; setter no-ops are confusing API, P2-05 — `FitResult.predict()` returns zeros (placeholder implementation), P2-06 — `MaskAssemble` history stack is publicly mutable, P2-07 — `UnifiedThreadingManager` reads `pool._max_workers` and `pool._threads` (private CPython attrs) (+7 more)

### Community 234 - "test_bayesian_assembly.py"
Cohesion: 0.16
Nodes (13): _identity_double_model(), _make_fit_result(), Tests for Bayesian assembly — converting per-Q FitResults to fit_summary., Create a FitResult with constant samples (mean = value, std ≈ 0)., Simplified double-exp model for testing., Test _extract_single_exp_params mapping., Test _extract_double_exp_params mapping., TestExtractDoubleExpParams (+5 more)

### Community 235 - "export_bayesian_csv"
Cohesion: 0.12
Nodes (14): Integration test: batch Bayesian -> dual storage -> plot -> export., Exported CSV must contain all Q-bins and parameters., export_bayesian_results must write CSV and figure files., Must write a CSV with parameter columns., CSV should include status column when failed_mask is provided., TestExportBayesianResults, export_bayesian_csv(), export_bayesian_diagnostics() (+6 more)

### Community 236 - "CommandPalette"
Cohesion: 0.13
Nodes (12): register_action should add action to palette., register_action should store shortcut., register_action should raise ValueError for duplicate ID., unregister_action should remove action., unregister_action should return False for unknown ID., Tests for action registration., TestCommandPaletteActions, CommandPalette (+4 more)

### Community 237 - "TestBug023FailedNLSQNaN"
Cohesion: 0.67
Nodes (3): BUG-023: failed NLSQResult metrics NaN, TestBug023FailedNLSQNaN, NLSQResult

### Community 238 - "ndimage.py"
Cohesion: 0.16
Nodes (19): scipy_replacements.__init__, _convolve_1d(), gaussian_filter(), gaussian_filter1d(), _gaussian_filter_1d_jax(), _gaussian_filter_jax(), Any, ArrayLike (+11 more)

### Community 239 - "TestQtTimerThreadingErrors"
Cohesion: 0.15
Nodes (8): Test timer creation in background thread (should fail)., Test proper Qt thread-based cleanup., Test Qt timer threading error detection., Test timer creation in main thread (should succeed)., Test detection of timer threading violations., Test proper Qt thread-based cleanup operations., Test background cleanup integration with error detection., TestQtTimerThreadingErrors

### Community 240 - "_create_hdf5_structure"
Cohesion: 0.16
Nodes (12): Group, _create_hdf5_structure(), create_temp_hdf5(), MockFactory, Any, Path, Factory for creating consistent mock objects for testing., Create mock XpcsFile object with realistic data structure. Args: data_type:… (+4 more)

### Community 241 - "TestBottleneck4C2Percentile"
Cohesion: 0.14
Nodes (12): c2_clean_baseline(), c2_clean_candidate(), make_c2_with_nans(), Current implementation: three separate sort passes (verbatim from twotime.py)., Proposed fix: one finite filtering pass and one triple-percentile call., Bottleneck #4 (baseline P1): clean_c2_for_visualization — 3 sort passes → 1., Candidate produces finite output consistent with baseline., Replacement values should be within data range. (+4 more)

### Community 242 - "test_cpu_only_launch.py"
Cohesion: 0.11
Nodes (11): Tests for CPU-only system launch (T068). Tests that application launches…, Tests for automatic CPU fallback when GPU fails., Test graceful fallback to CPU if GPU initialization fails., Test computations succeed after GPU fallback., Tests for CPU-only system launch., Test backend initializes when GPU is disabled., Test fallback to NumPy when JAX is explicitly disabled., Test Q-map computation works on CPU. (+3 more)

### Community 243 - "test_plot_themes.py"
Cohesion: 0.11
Nodes (11): Unit tests for plot_themes module., Light theme should return light-colored matplotlib params., Dark theme should return dark-colored matplotlib params., Matplotlib params should have all required rcParams keys., All matplotlib param values should be valid hex colors., Tests for Matplotlib theme parameter generation., Tests for PyQtGraph theme integration., apply_to_pyqtgraph should not raise any errors. (+3 more)

### Community 244 - "RecentPathsManager"
Cohesion: 0.15
Nodes (12): RecentPathsManager should have default max_entries=10., RecentPathsManager should accept custom max_entries., Tests for RecentPathsManager persistence., Manager should load existing recent_paths.json., Manager should persist data to file., Manager should handle corrupted JSON gracefully., Tests for RecentPathsManager initialization., RecentPathsManager should be created. (+4 more)

### Community 245 - "MockH5pyGroup"
Cohesion: 0.13
Nodes (6): MockH5pyDataset, MockH5pyGroup, ndarray, Mock h5py Dataset for testing., Mock h5py Group for testing., Create a mock dataset.

### Community 246 - "ToastManager"
Cohesion: 0.16
Nodes (9): Manages toast notifications for a parent window. Handles creating, displaying,…, Show a toast notification. Args: message: Text to display toast_type: Type…, Show a success toast., Show a warning toast., Dismiss all visible toasts immediately., Set the default duration for toasts. Args: duration_ms: Default duration in…, Dismiss a specific toast., Position all toasts in bottom-right corner. (+1 more)

### Community 247 - "TestTwoTimeMatrixOperations"
Cohesion: 0.12
Nodes (9): Test mathematical operations on two-time correlation matrices, Create correlation matrix with exponential time dependence, Test trace properties of correlation matrices, Test determinant properties indicating matrix regularity, Test condition number for numerical stability, Test properties of inverse correlation matrix, Test PCA decomposition of correlation matrix, Test filtering operations on correlation matrices (+1 more)

### Community 248 - ".run_comprehensive_validation"
Cohesion: 0.23
Nodes (7): Validate statistical calculations against theoretical distributions., Validate Fourier transform calculations., Result of a scientific validation test., Validate q-space and scattering vector calculations., Validate intensity normalization procedures., Run all validation tests and return comprehensive results., ValidationResult

### Community 249 - "test_hotpath_baseline.py"
Cohesion: 0.16
Nodes (16): c2_matrix(), fft_data(), fitting_data(), g2_data(), fixture, qmap_geometry(), qmap_geometry_large(), Comprehensive baseline benchmarks for all 6 computational hot paths. Covers: 1.… (+8 more)

### Community 250 - "TestJITWarmup"
Cohesion: 0.12
Nodes (11): skipif, Tests for JIT compilation warmup (T055). Tests that JIT compilation triggers on…, Tests for Q-map JIT compilation warmup., Test JIT-compiled Q-map function exists., Test Q-map first call includes compilation., Tests for JIT compilation warmup behavior., Test first call triggers JIT compilation (slower)., Test JIT caches compiled function for reuse. (+3 more)

### Community 251 - "Python-Level Optimization Report"
Cohesion: 0.12
Nodes (16): 1.1 Early Cache Exit in `_load_saxs_data_batch()`, 1.2 Reuse Cached HDF5 Reader Singleton in `load_data()`, 1.3 Module-Level `hashlib` Import, 1. HDF5 I/O Pipeline (Task #5), 2.1 MaskAssemble: Read-Only View (`_get_mask_ref()`), 2.2 Bounded Mask History, 2.3 `compute_g2_ensemble_statistics`: Opt-In `np.median`, 2.4 `batch_g2_normalization`: Vectorized Stack Path (+8 more)

### Community 252 - ".on_async_plot_ready"
Cohesion: 0.13
Nodes (7): Handle async plot completion., Apply SAXS 2D plot result to the GUI., Apply G2 plot result to the GUI. Uses the pre-computed data returned by the…, Apply two-time plot result to the GUI., Apply intensity plot result to the GUI., Apply stability plot result to the GUI., Apply Q-map plot result to the GUI.

### Community 253 - "SoftwarePackageValidator"
Cohesion: 0.15
Nodes (12): create_reference_comparison_report(), load_reference_data_from_file(), Any, ndarray, Validator using reference results from established XPCS software packages, Validate against reference software package results Args: input_data: Must…, Compare data from a specific field, MATLAB DLS toolbox specific validation (+4 more)

### Community 254 - "tests.scientific (package)"
Cohesion: 0.26
Nodes (13): tests.scientific.reference_validation (package), tests.scientific.algorithms (package), test_fitting_algorithms, SCIENTIFIC_CONSTANTS, VALIDATION_CONFIG, tests.scientific (package), tests.scientific.properties (package), mathematical_invariants (+5 more)

### Community 255 - "ShortcutManager"
Cohesion: 0.14
Nodes (11): Tests for ShortcutManager initialization., ShortcutManager should be created., ShortcutManager should have shortcut_triggered signal., ShortcutManager should start with no shortcuts., TestShortcutManagerInit, QObject, Get list of registered shortcut IDs. Returns: List of shortcut identifiers, Check if a shortcut ID is registered. Args: shortcut_id: ID to check Returns:… (+3 more)

### Community 256 - "Subsystem Responsibilities"
Cohesion: 0.15
Nodes (12): Analysis Modules, Backend Abstraction, Core Data Model, Fitting Pipeline, GUI Layer, I/O Layer, Module Dependency Diagram, Schema Validation (+4 more)

### Community 257 - "test_g2_partial_safety.py"
Cohesion: 0.13
Nodes (11): scientific, Unit tests for g2_partial Safety Check (Feature 3). Tests the safety check in…, Tests for g2_partial attribute handling in XpcsFile., XpcsFile.g2_partial should return None when data is not in HDF5., g2_partial lazy loading should handle exceptions gracefully., Scientific validation tests for g2_partial data handling., g2_partial should have correct shape (frames, delays, qbins)., g2_partial values should be in physical range (0 < g2 < 2 typically). (+3 more)

### Community 258 - "test_gixpcs_precision.py"
Cohesion: 0.12
Nodes (10): fixture, Unit tests for GIXPCS Display Precision (Feature 4). Tests the enhanced…, Create a minimal mock QMap object., Create mock QMap with all relevant keys., Integration tests with the actual QMap class., Create a minimal QMap-like object for testing., Verify complete output format with mixed precision., Verify units are preserved in output. (+2 more)

### Community 259 - "Any"
Cohesion: 0.12
Nodes (9): Any, Create zero-filled array with same shape as input., Create ones-filled array with same shape as input., Create array filled with specified value., Create array from data., Cast array to specified dtype., Create array filled with zeros., Create array filled with ones. (+1 more)

### Community 260 - "XPCS Viewer Dependency Diagram"
Cohesion: 0.12
Nodes (15): Anti-Patterns Identified, Circular Dependency Check (None Found ✅), Complexity Metrics, Critical I/O Boundaries (Conversion Points), Data Flow: SimpleMask Integration, Data Flow: XPCS Analysis Pipeline, Layer 1: Backend Abstraction (Foundation), Layer 2: Data Access & I/O (+7 more)

### Community 261 - "create_slice"
Cohesion: 0.09
Nodes (23): Tests for type annotations in helper/utils.py (Task T064). This test verifies…, Verify norm_saxs_data returns correct tuple., Verify create_slice returns a slice object., Verify helper/utils.py functions have complete type annotations., T064: Verify all functions have return and parameter type annotations., T060: Verify get_min_max has correct type signature., T061: Verify norm_saxs_data has correct type signature., T062: Verify create_slice has correct type signature. (+15 more)

### Community 262 - "ListDataModel"
Cohesion: 0.14
Nodes (3): mock_viewer_kernel(), Create a mock ViewerKernel instance., ListDataModel

### Community 263 - "MockH5py"
Cohesion: 0.14
Nodes (11): Tests for GUI error handling and edge cases. This module provides comprehensive…, Test suite for concurrency-related issues., Test for thread safety issues in GUI updates., Test suite for resource cleanup and memory leaks., Test that plot resources are properly cleaned up., Test that file handles are properly closed., TestConcurrencyIssues, TestResourceCleanup (+3 more)

### Community 264 - "TestAPS8IDIPathFormats"
Cohesion: 0.12
Nodes (9): Test suite for APS 8IDI path format validation., Test that all path values are strings., Test that most paths start with '/' (absolute paths)., Test that qmap paths follow consistent structure., Test that multitau paths follow consistent structure., Test that two-time paths follow consistent structure., Test that temporal mean paths follow consistent structure., Test that entry-level paths follow consistent structure. (+1 more)

### Community 265 - "QtThreadingValidator"
Cohesion: 0.12
Nodes (11): QtThreadingValidator, Validate Qt threading compliance., Check if running in main Qt thread., Validate that timer is created in appropriate thread context., Validate signal/slot connection syntax., Test Qt error detection framework., Test Qt error capture initialization., Test main thread detection. (+3 more)

### Community 266 - "ADR-003: HDF5 Facade Pattern with Connection Pooling"
Cohesion: 0.17
Nodes (11): ADR-003: HDF5 Facade Pattern with Connection Pooling, Architecture, Connection Pooling, Consequences, Context, Decision, Facade Design, Schema Design (+3 more)

### Community 267 - "TestLogTiming"
Cohesion: 0.12
Nodes (9): Tests for log_timing decorator., log_timing logs the execution time., log_timing logs at the specified level., log_timing elevates level when threshold exceeded., log_timing includes function arguments when include_args=True., log_timing logs error and re-raises on exception., log_timing preserves the decorated function's metadata., log_timing returns the original function's result. (+1 more)

### Community 269 - "qt_threading_utils.py"
Cohesion: 0.15
Nodes (11): create_thread_safe_timer(), QtThreadSafetyValidator, Qt Threading Violation Detection Utilities. This module provides utilities for…, Validate Qt thread safety patterns., Validate that QTimer is being used safely. Args: timer_obj: QTimer instance to…, Validate signal/slot connection for Qt5+ compatibility. Args: signal: Signal…, Validate that GUI operations are happening in the main thread. Args:…, Test signal/slot connections across threads. (+3 more)

### Community 270 - "LazyMplCanvasBarV"
Cohesion: 0.14
Nodes (7): __getattr__(), PlaceholderClass, Plot rendering backends for XPCS visualization. This package provides theme-…, Placeholder class for documentation builds., Lazily load matplotlib and pyqtgraph backends to improve startup time., LazyMplCanvasBarV, QWidget

### Community 271 - "Reference Data for Scientific Validation"
Cohesion: 0.13
Nodes (14): Adding New Reference Data, Data Format, `/diffusion_analysis/`, Directory Structure, External References, `/g2_analysis/`, Maintenance, Metadata Fields (+6 more)

### Community 272 - "get_project_root"
Cohesion: 0.26
Nodes (14): generate_report(), get_project_root(), main(), Run code quality checks., Run test coverage analysis., Generate validation report., Get the project root directory., Main validation runner. (+6 more)

### Community 273 - "framework/utils.py"
Cohesion: 0.13
Nodes (14): _get_memory_usage(), gui_test(), MockH5py, performance_test(), Test utilities and helper functions for XPCS Toolkit tests. This module…, Decorator for GUI tests. Args: requires_display: Whether test requires actual…, Context manager to suppress specific warning types during testing. Usage: with…, Get current memory usage in MB. (+6 more)

### Community 274 - "TestFittingAlgorithmProperties"
Cohesion: 0.16
Nodes (10): Any, given, settings, Property-based test for fitting robustness to noise, Test that fitting results are independent of initial conditions, Test general properties that all fitting algorithms should satisfy, Set up common test parameters, Create test function with known parameters (+2 more)

### Community 275 - "plot_bayesian_all_q"
Cohesion: 0.17
Nodes (10): Tests for Bayesian all-Q visualization and export., Must return a matplotlib Figure with axes., Must return None when bayesian_summary is None., LineCollection should exclude Q-bins flagged as failed., plot_bayesian_all_q must generate a matplotlib figure., Title should include success count when failed_mask is present., TestPlotBayesianAllQ, plot_bayesian_all_q() (+2 more)

### Community 276 - "test_tab_availability.py"
Cohesion: 0.24
Nodes (10): gui, unit, Routing checks for XpcsViewer.update_tab_availability. Exercises the format-…, Minimal stand-in for the parts of XpcsViewer the method touches., _Recorder, _run(), test_both_file_enables_all_tabs(), test_mixed_single_type_files_enable_all_tabs() (+2 more)

### Community 277 - "MemoryTestUtils"
Cohesion: 0.15
Nodes (10): Test performance of Qt error detection framework., Test performance of error capture mechanism., Test memory usage during error detection., TestQtErrorDetectionPerformance, MemoryTestUtils, Memory Testing Utilities. Minimal implementation to support Qt error detection…, Memory testing utilities for test framework., Get current memory usage in bytes. (+2 more)

### Community 278 - ".load_dataset"
Cohesion: 0.18
Nodes (10): _decode_unit(), _log_qmap_shape(), _normalize_unit(), Log Q-map array shapes at DEBUG level., Normalize num_pts to be a 2-element array [n_dim0, n_dim1]. Args: num_pts: Can…, Get default values for missing qmap keys., Create a minimal default qmap when file doesn't have qmap data., Provide dictionary-like access to QMap attributes. (+2 more)

### Community 279 - "ProgressDialog"
Cohesion: 0.15
Nodes (9): ProgressDialog, QDialog, Mark the operation as completed., Mark the operation as cancelled., Dialog for showing progress of multiple operations., Mark an operation as completed., Mark an operation as cancelled in the UI., Remove a completed operation from the dialog. (+1 more)

### Community 280 - "ProgressManager"
Cohesion: 0.14
Nodes (9): QStatusBar, ProgressManager, QObject, Centralized progress management for the application., Set the main window status bar for simple progress display., Mark an operation as completed., Check if an operation is currently active., Get set of active operation IDs. (+1 more)

### Community 281 - "TestGetData"
Cohesion: 0.14
Nodes (8): Test data retrieval with files that don't have correlation analysis., Test data retrieval with mixed analysis types., Test data retrieval with Twotime analysis type., Test data retrieval with None ranges., Test data retrieval with empty file list., Test suite for get_data function., Test successful data retrieval., TestGetData

### Community 282 - "TestComputeGeometry"
Cohesion: 0.14
Nodes (8): Test suite for compute_geometry function., Test geometry computation for 'multiple' plot type., Test geometry computation for 'single' plot type., Test geometry computation for 'single-combined' plot type., Test geometry computation with invalid plot type., Test geometry computation with single file., Test geometry computation uses first file's shape., TestComputeGeometry

### Community 283 - "PerformanceTimer"
Cohesion: 0.14
Nodes (6): PerformanceTimer, Context manager for measuring execution time with statistical analysis., Standard deviation of execution time., Minimum execution time., Maximum execution time., Generate performance report.

### Community 284 - "ComprehensiveCrossValidationFramework"
Cohesion: 0.19
Nodes (9): AnalyticalValidator, ComprehensiveCrossValidationFramework, Comprehensive framework combining multiple validation approaches, Add a validator to the framework, Validator that compares against analytical solutions, Test cases for the cross-validation framework, Set up test framework, Test analytical validation with noisy data (+1 more)

### Community 285 - "test_bayesian_dual_storage.py"
Cohesion: 0.14
Nodes (9): Tests for dual fit_summary storage on XpcsFile., bayesian_fit_summary starts as None., Setting bayesian_fit_summary must not affect fit_summary., assemble_fit_summary must include source='bayesian' in output., Assembled fit_summary must have source='bayesian'., q_range and t_range args must appear in output., XpcsFile must store NLSQ and Bayesian fit summaries independently., TestAssembleFitSummarySource (+1 more)

### Community 286 - "TestBug025FitDiagnosticsValidation"
Cohesion: 0.14
Nodes (8): BUG-025: FitDiagnostics.__post_init__ must reject invalid diagnostics., FitDiagnostics with negative divergences must raise ValueError., FitDiagnostics with negative ESS bulk values must raise ValueError., FitDiagnostics with negative ESS tail values must raise ValueError., FitDiagnostics with valid values must be accepted., Zero divergences must be accepted (boundary case)., Zero ESS values must be accepted (warm-up only case)., TestBug025FitDiagnosticsValidation

### Community 287 - ".update_tab_availability"
Cohesion: 0.24
Nodes (4): Update tab availability based on the file formats in the target list. Disables…, Configure tabs for multitau format: disable Two Time tab., Configure tabs for twotime format: disable G2, G2 Fitting, G2 Map, and…, Clear all tab tooltips.

### Community 288 - "ADR-002: Migration from scipy.optimize to NLSQ 0.6.0"
Cohesion: 0.18
Nodes (10): ADR-002: Migration from scipy.optimize to NLSQ 0.6.0, Architecture, Consequences, Context, Decision, Key Design Choices, Statistical Properties (Delegated from CurveFitResult), Status (+2 more)

### Community 289 - "CommandAction"
Cohesion: 0.19
Nodes (9): Unit tests for CommandPalette., Tests for CommandAction dataclass., CommandAction should store all fields., CommandAction should store shortcut., CommandAction should store enabled callable., TestCommandAction, CommandAction, Register an action with the command palette. Args: action_id: Unique identifier… (+1 more)

### Community 290 - "XPCS Viewer Documentation Structure"
Cohesion: 0.18
Nodes (10): API Reference (Information-Oriented), Building Documentation, Diataxis Framework, Documentation Tree, Explanation (Understanding-Oriented), How-To Guides (Task-Oriented), Key Features, Shared Fragments (+2 more)

### Community 291 - "TestDragDropListViewMoveItem"
Cohesion: 0.14
Nodes (8): move_item should return False for invalid target index., move_item should return False when source equals target., move_item should successfully reorder items., move_item should emit items_reordered signal on success., Tests for programmatic item movement., move_item should return False when no model., move_item should return False for invalid source index., TestDragDropListViewMoveItem

### Community 292 - "tests/utils/reliability.py"
Cohesion: 0.07
Nodes (28): Lock, FlakinessDetector, get_flakiness_detector(), get_resource_lock_manager(), Any, Manage resource locks to prevent test interference., Get or create a lock for a resource., Acquire exclusive access to a resource. (+20 more)

### Community 294 - "1. P0 — Critical: Crash, Data Corruption, or Silent Wrong Result"
Cohesion: 0.18
Nodes (11): 1. P0 — Critical: Crash, Data Corruption, or Silent Wrong Result, P0-01 — Missing `extra_fields=("diverging",)` in MCMC → KeyError after sampling, P0-02 — `__dict__` access on frozen dataclass `QMapSchema` → AttributeError, P0-03 — `double_exp_model` NaN gradients: `tau1 ≈ 0` → `exp(0/0)`, P0-04 — `read_g2_data` does not coerce float32 → float64 before `G2Data` constructor, P0-05 — `read_qmap` does not coerce float32 → float64 before `QMapSchema` construction, P0-06 — Qt signals emitted from raw `threading.Thread` in `_monitor_system`, P0-07 — `@Slot` decorators missing on `invokeMethod` targets → signals silently dropped (+3 more)

### Community 295 - "TestPlotThemesModule"
Cohesion: 0.14
Nodes (8): Tests for standalone plot_themes module functions., MATPLOTLIB_LIGHT dict should exist and have proper structure., MATPLOTLIB_DARK dict should exist and have proper structure., Light and dark params should have different values., get_plot_colors should return colors dict for light theme., get_plot_colors should return colors dict for dark theme., get_pyqtgraph_options should return config dict., TestPlotThemesModule

### Community 296 - "test_qss_lint.py"
Cohesion: 0.18
Nodes (11): _extract_selectors(), parametrize, Path, Lint tests for QSS stylesheets. Catches duplicate selectors and unsupported CSS…, Detect CSS properties not supported by Qt's QSS engine., QSS files should not use outline, outline-offset, or letter-spacing., Extract (selector, line_number) pairs from QSS text. Returns normalised…, Detect duplicate selectors within a single QSS file. (+3 more)

### Community 297 - "test_session_field_completeness.py"
Cohesion: 0.19
Nodes (10): _get_analysis_param_fields(), _get_collect_state_keywords(), Test that _collect_session_state covers all AnalysisParameters fields. Uses AST…, Return all field names from AnalysisParameters dataclass., Parse _collect_session_state and return keyword arg names used in the…, Ensure _collect_session_state maps all AnalysisParameters fields., Every AnalysisParameters field must appear as a keyword arg in…, _NO_WIDGET_FIELDS should not contain names that don't exist in… (+2 more)

### Community 298 - "TestShortcutRegistration"
Cohesion: 0.14
Nodes (8): unregister_shortcut should return False for unknown ID., Tests for shortcut registration., register_shortcut should accept string key sequence., register_shortcut should accept QKeySequence., register_shortcut should return False for duplicate ID., register_shortcut should return False without parent., unregister_shortcut should remove shortcut., TestShortcutRegistration

### Community 299 - "test_tg3_mask_export_and_g2_plot.py"
Cohesion: 0.21
Nodes (13): _make_viewer(), _make_xf(), Unit tests for TG3 — GUI Critical Fixes: Mask Export and Plot Data. Tests for…, check_g2_number must assign val = default_val[n] in the except block. When a…, Exported mask must be visible on all XpcsFiles before update_plot is called.…, Return a minimal XpcsFile-like mock., Return a minimal XpcsViewer-like mock with the methods under test. We import…, import_mask must set xf.mask on every loaded XpcsFile and call update_plot. (+5 more)

### Community 300 - "TestThemeManagerTokenAccess"
Cohesion: 0.14
Nodes (8): get_color should raise KeyError for invalid tokens., get_spacing should return valid spacing for known sizes., get_spacing should raise KeyError for invalid sizes., get_tokens should return current ThemeDefinition., Tests for token access methods., get_color should return valid color for known tokens., get_color should return dark theme colors when dark., TestThemeManagerTokenAccess

### Community 301 - "Backend Abstraction Pattern"
Cohesion: 0.10
Nodes (22): Backend Abstraction Layer (ADR-004), Facade Infrastructure, HDF5 Facade Pattern (ADR-003), JAX Migration (ADR-001), Backend Abstraction Pattern, HDF5 Connection Pooling, HDF5 Data Format for XPCS, I/O Boundary Array Conversion (+14 more)

### Community 302 - "Data Flow"
Cohesion: 0.20
Nodes (9): Array Type Transitions, Convergence Thresholds, Data Flow, Fitting Data Flow, HDF5 File Schema, I/O Boundary Summary, Signal Payload Schemas, SimpleMask Data Flow (+1 more)

### Community 303 - "TestGetShortcutMap"
Cohesion: 0.19
Nodes (9): Tests for get_shortcut_map function., Shortcut map should not be empty., Shortcut map should contain all drawing tool shortcuts., Shortcut map should contain eraser shortcut., All shortcut keys should be strings., All shortcut values should be tool name strings., TestGetShortcutMap, get_shortcut_map() (+1 more)

### Community 304 - "test_partition.py"
Cohesion: 0.14
Nodes (9): Unit tests for SimpleMask partition (Q-binning) functionality. Tests the…, Tests for partition overlay display., Window should track partition overlay state., Partition toggle should require computed partition., Tests for partition export functionality., export_partition_to_viewer method should exist., qmap_exported signal should emit partition dict., TestPartitionExport (+1 more)

### Community 305 - "TestEnsureNumpyAtPyQtGraphBoundaries"
Cohesion: 0.14
Nodes (8): BUG-027: All JAX arrays must be converted via ensure_numpy() before reaching…, ensure_numpy must accept JAX arrays and return NumPy arrays., ensure_numpy must be a no-op for NumPy arrays., apply_saxs_2d_result in xpcs_viewer.py must wrap image_data with ensure_numpy., apply_qmap_result in xpcs_viewer.py must wrap image_data with ensure_numpy., simplemask_kernel.py refresh_detector_image must use ensure_numpy., simplemask_window.py _refresh_mask_display must use ensure_numpy., TestEnsureNumpyAtPyQtGraphBoundaries

### Community 306 - ".capture_qt_warnings"
Cohesion: 0.12
Nodes (10): _CaptureContext, Test Qt signal/slot connection error detection., Test detection of QStyleHints connection errors., Test proper Qt5+ signal/slot connection syntax., Test Qt GUI initialization error detection., Context manager to capture Qt warnings., Test monitoring of Qt application creation., Test widget creation in proper Qt context. (+2 more)

### Community 307 - "ndarray"
Cohesion: 0.15
Nodes (9): ArrayLike, ndarray, Model predictions at input x values., Set predictions (for backward compat initialization). Note: This setter is a…, Parameter covariance matrix., Set covariance (for backward compat initialization). Note: This setter is a no-…, Get prediction interval at new x values. Delegates to…, Get posterior samples for parameter. Parameters ---------- param : str… (+1 more)

### Community 308 - "ObjectRegistry"
Cohesion: 0.16
Nodes (8): ObjectRegistry, Any, Registry for tracking objects that need cleanup., Register an object for tracking., Unregister an object., Get a registered object., Clear all registered objects, calling close() on each (P1-16)., Get all registered objects of a specific type. Args: obj_type: Type name to…

### Community 309 - "HealthStatus"
Cohesion: 0.16
Nodes (9): HealthStatus, Overall system health status levels., Check metrics and trigger alerts if thresholds exceeded., Trigger registered callbacks for overall status. Warning: Callbacks are invoked…, Register callback for specific health status. Warning: Callbacks are invoked on…, Calculate overall health status., Register callback for specific health status changes., Get current status based on thresholds. (+1 more)

### Community 310 - "._monitoring_loop"
Cohesion: 0.16
Nodes (7): Main monitoring loop running in background thread., Update system-level metrics using psutil., Update application-specific metrics., Update garbage collection metrics. ``initial_stats`` must be the GC stats…, Get current HDF5 connection count., Clean up dead weak references., Update metric value and history.

### Community 311 - "TestPgPlotFunction"
Cohesion: 0.15
Nodes (9): patch, Test suite for pg_plot function (basic structure tests)., Test that pg_plot function exists and is callable., Test basic call structure of pg_plot function., Test suite for pg_plot function parameters., Test pg_plot function parameter defaults., Test that pg_plot has expected parameter names., TestPgPlotFunction (+1 more)

### Community 312 - ".evaluate_benchmark"
Cohesion: 0.19
Nodes (8): create_custom_analytical_benchmark(), Any, Analytical Benchmarks for XPCS Algorithm Validation This module provides…, Get benchmark data by name, Evaluate a benchmark with given or default parameters Args: benchmark_name:…, Validate that benchmark results satisfy expected properties Args:…, Run validation on all benchmarks with their default parameters Returns:…, Create a custom analytical benchmark Args: function: Analytical function to use…

### Community 313 - ".run_comprehensive_validation"
Cohesion: 0.17
Nodes (7): Any, Validate against reference software implementation Args: input_data: Contains…, Validate algorithm output against reference implementation Args: input_data:…, Run comprehensive validation across multiple test cases and validators Args:…, Generate a validation certificate summarizing results, Test analytical validation for G2 single exponential, Validate against analytical solutions Args: input_data: Must contain…

### Community 314 - "LiteratureReferenceValidator"
Cohesion: 0.21
Nodes (7): LiteratureReferenceValidator, Validate sphere form factor against literature, Validate G2 fitting parameters against literature, Validator using reference data from scientific literature, Validate two-time analysis against literature, Load literature reference data, Validate against literature reference Args: input_data: Must contain…

### Community 315 - "TestRecentPathsManagerAddPath"
Cohesion: 0.15
Nodes (8): fixture, Tests for RecentPathsManager.add_path., Create manager with temporary storage., add_path should add path to recent list., add_path should update existing path., add_path should move existing path to front., add_path should trim list to max_entries., TestRecentPathsManagerAddPath

### Community 316 - "test_qt_error_detection.py"
Cohesion: 0.19
Nodes (12): background_cleanup_tester(), BackgroundCleanupTester, fixture, qapp(), qt_error_capture(), qt_threading_validator(), Qt Error Detection Test Framework. This module provides comprehensive testing…, Test background cleanup operations for Qt compliance. (+4 more)

### Community 317 - "TestSaxsBaseline"
Cohesion: 0.17
Nodes (7): Baseline benchmarks for SAXS 1D processing (hot path #4)., Verify vectorized q-binning produces valid output., Benchmark vectorized q-binning (1000 points, 100 bins)., Benchmark vectorized background subtraction., Benchmark batch SAXS analysis with normalize+trim operations., Stress test: q-binning on large flattened detector (512x512)., TestSaxsBaseline

### Community 318 - "MockH5pyFile"
Cohesion: 0.17
Nodes (5): mock_hdf5_file(), MockH5pyFile, Create a temporary HDF5 file with mock XPCS data., Create temporary HDF5 files for testing., temp_hdf5_files()

### Community 319 - "TestSimpleMaskFromViewer"
Cohesion: 0.17
Nodes (7): Tests for launching SimpleMask from XPCS Viewer., Mask Editor should be accessible from XPCS Viewer toolbar., open_simplemask should create SimpleMask window., open_simplemask should show the window., Calling open_simplemask twice should reuse the window., SimpleMask window should have reference to parent viewer., TestSimpleMaskFromViewer

### Community 320 - "tests.unit (package)"
Cohesion: 0.20
Nodes (10): tests.unit (package), test_lazy_loading, test_package_basics, test_plot_constants, test_qmap_constants, test_tg4_backend_io_fixes, test_tg7_gui_io_p1_fixes, test_tg8_code_quality (+2 more)

### Community 321 - "test_qt_compat.py"
Cohesion: 0.17
Nodes (8): Tests for Qt compatibility layer (User Story 4). These tests verify the…, T059: Verify qt_compat defaults to pyside6 when QT_API is unset., T059: Verify qt_compat respects QT_API environment variable., Test qt_compat with PySide6 backend (T057). Note: These tests require PySide6…, T057: Verify application works with PySide6 (default)., Test QT_API environment variable handling., TestQtApiEnvironment, TestQtCompatWithPySide6

### Community 322 - "TestQtCompatLayer"
Cohesion: 0.17
Nodes (7): Test qt_compat module provides correct abstraction., T057: Verify qt_compat exports core Qt modules., T057: Verify Signal and Slot are exported., T057: Verify common widget classes are exported., T057: Verify GUI-related classes are exported., T057: Verify threading-related classes are exported., TestQtCompatLayer

### Community 323 - "test_fitting_algorithms.py"
Cohesion: 0.17
Nodes (7): Algorithm-specific validation tests This module contains comprehensive…, Fitting Algorithm Validation Tests This module provides comprehensive…, Test statistical properties of fitting algorithms, Test analysis of parameter correlations in fitting, Test that confidence intervals have correct coverage probability, Test statistical properties of fitting residuals, TestFittingStatisticalValidation

### Community 324 - "cross_validation_framework.py"
Cohesion: 0.23
Nodes (8): ABC, Cross-Validation Framework for XPCS Algorithms This module provides a…, Abstract base class for reference validation, Validator that compares against reference software implementations, ReferenceImplementationValidator, ReferenceValidator, Reference Validation and Cross-Validation Framework This module provides cross-…, Reference Implementation Validators This module provides validation against…

### Community 327 - "TestCommandPaletteSearch"
Cohesion: 0.17
Nodes (7): Tests for search/filter functionality., set_placeholder should update search input., _fuzzy_match should match prefix., _fuzzy_match should match substring., _fuzzy_match should match word initials., _fuzzy_match should return False for no match., TestCommandPaletteSearch

### Community 328 - "TestCommandPaletteInit"
Cohesion: 0.17
Nodes (7): Tests for CommandPalette initialization., CommandPalette should be created., CommandPalette should have search input., CommandPalette should have results list., CommandPalette should have action_triggered signal., CommandPalette should have correct objectName., TestCommandPaletteInit

### Community 329 - "test_theme_manager.py"
Cohesion: 0.17
Nodes (8): Unit tests for ThemeManager., Tests for Matplotlib integration., get_matplotlib_params should return valid rcParams dict., Matplotlib params should be different for light vs dark., Tests for PyQtGraph integration., apply_to_pyqtgraph should not raise errors., TestThemeManagerMatplotlib, TestThemeManagerPyQtGraph

### Community 330 - "._on_item_activated"
Cohesion: 0.22
Nodes (5): QListWidgetItem, Hide the command palette dialog., Execute the currently selected action., Handle action item activation., Handle key press events.

### Community 331 - "XPCS Viewer — Master Fix List"
Cohesion: 0.22
Nodes (8): 5. Recommended Fix Order (Dependency-Aware), How to Read This Report, Wave 1 — Unblock core functionality (fix before any testing), Wave 2 — Fix crash/data-corruption risks under normal use, Wave 3 — Fix reliability and UI correctness issues, Wave 4 — Fix UX and session state issues, Wave 5 — Technical debt and API hygiene (can be batched as a refactor sprint), XPCS Viewer — Master Fix List

### Community 332 - ".validate_fitting_algorithms"
Cohesion: 0.20
Nodes (6): ndarray, Validate fitting algorithm accuracy against known functions., Fit single exponential function (mock implementation)., Fit double exponential function (mock implementation)., Subtract background from intensities (mock implementation)., Test fitting algorithm validation methods.

### Community 333 - "TestThemeManagerBasics"
Cohesion: 0.17
Nodes (7): Basic ThemeManager tests., ThemeManager should default to system theme detection., Setting light theme should update current theme., Setting dark theme should update current theme., Theme changes should emit theme_changed signal., No signal should be emitted if theme doesn't actually change., TestThemeManagerBasics

### Community 334 - "test_interpolation.py"
Cohesion: 0.21
Nodes (8): Tests for JAX-native interpolation migration. Tests for Technical Guidelines…, T043: Test optimized_c2_sampling with JAX-based zoom replacement., Verify bilinear downsampling produces correct output size., Verify uniform downsampling produces correct output., Verify no change when target_size >= current_size., TestOptimizedC2Sampling, optimized_c2_sampling(), Optimized C2 matrix downsampling using vectorized operations. Args: c2_matrix:…

### Community 335 - "constants.__init__"
Cohesion: 0.25
Nodes (8): constants.defaults, constants.fitting, constants.__init__, constants.limits, constants.thresholds, constants.timeouts, aps_8idi key map, HDF5Reader

### Community 336 - "TestEraserTool"
Cohesion: 0.17
Nodes (7): Tests for ERASER_TOOL definition., ERASER_TOOL should be defined., Eraser tool should have correct name., Eraser tool should use Rectangle type., Eraser tool should use inclusive mode., Eraser tool should have keyboard shortcut., TestEraserTool

### Community 337 - "XPCS Viewer Dependency Analysis and Integration Catalog"
Cohesion: 0.25
Nodes (7): 8. Risk Assessment, Dependency Health Overview, Executive Summary, High-Risk Integration Points, Key Findings, Mitigation Strategies, XPCS Viewer Dependency Analysis and Integration Catalog

### Community 338 - "5.1 Critical Facades Needed"
Cohesion: 0.25
Nodes (8): 1. HDF5 I/O Facade (Priority: HIGH), 2. Backend Array Adapter (Priority: MEDIUM), 3. Data Schema Validators (Priority: HIGH), 5.1 Critical Facades Needed, 5.2 Circular Dependency Risks, 5.3 Tight Coupling Analysis, 5. Integration Points Requiring Attention, High Coupling: XpcsFile ↔ Analysis Modules

### Community 339 - "DrawingTool"
Cohesion: 0.23
Nodes (8): Tests for DrawingTool dataclass., DrawingTool should be creatable with required fields., DrawingTool should default to exclusive mode., DrawingTool should default to CrossCursor., DrawingTool should accept inclusive mode., TestDrawingToolDataclass, DrawingTool, Configuration for a drawing tool. Attributes: name: Tool display name…

### Community 340 - "TestGIXPCSScientificValidation"
Cohesion: 0.21
Nodes (8): parametrize, scientific, Scientific validation tests for GIXPCS precision requirements., GIXPCS qx values are typically very small, requiring high precision., GIXPCS qr values require high precision for proper analysis., Standard transmission XPCS q values are fine with 3 decimals., Verify qx formatting maintains expected decimal places., TestGIXPCSScientificValidation

### Community 341 - "BackgroundThreadTester"
Cohesion: 0.17
Nodes (7): BackgroundThreadTester, Test background thread compliance with Qt requirements., Test timer creation in background thread (should fail)., Test timer creation in proper QThread (should succeed)., Run all background thread compliance tests., Generate compliance report from test results., test()

### Community 342 - "TestEnvironmentValidator"
Cohesion: 0.23
Nodes (7): Validate test environment prerequisites., Check if sufficient memory is available., Check if sufficient disk space is available., Check if network connectivity is available., Check if display is available for GUI tests., Validate multiple environment requirements., TestEnvironmentValidator

### Community 343 - "constants/__init__.py"
Cohesion: 0.17
Nodes (6): Default configuration values for xpcsviewer. These constants define default…, Fitting and model parameter constants for xpcsviewer. These constants define…, Centralized constants module for xpcsviewer. This module provides application-…, Size and count limit constants for xpcsviewer. These constants define maximum…, Numeric threshold constants for xpcsviewer. These constants define comparison…, Timeout constants for xpcsviewer operations. These constants define time limits…

### Community 344 - "HealthMetric"
Cohesion: 0.20
Nodes (7): HealthMetric, Trigger alert for specific metric., Handle specific metric alerts with automatic recovery actions., Perform emergency memory cleanup., Clean up HDF5 connections., Individual health metric with thresholds and history., Get trend over specified time window.

### Community 345 - "TestG2VectorizedOperations"
Cohesion: 0.18
Nodes (6): Test vectorized G2 operations for correctness and performance, Test vectorized baseline correction, Test batch normalization methods, Test ensemble statistics computation, Test error propagation for G2 operations, TestG2VectorizedOperations

### Community 346 - "ScientificAssertions"
Cohesion: 0.22
Nodes (7): ndarray, Helper class providing scientific assertion methods with proper tolerances., Assert that two arrays are element-wise close within tolerances. Args: actual:…, Assert that correlation function data satisfies physical constraints. Args:…, Assert that scattering data satisfies physical constraints. Args: q: Scattering…, Assert that fit quality metrics are reasonable. Args: chi_squared: Chi-squared…, ScientificAssertions

### Community 347 - "gui"
Cohesion: 0.28
Nodes (6): gui, Test suite for SAXS 2D analysis tab., Test SAXS 2D tab initializes with proper components., Test SAXS 2D image display functionality., Test SAXS 2D colorbar and scaling controls., TestSAXS2DTab

### Community 348 - "MockH5pyFile"
Cohesion: 0.20
Nodes (3): MockH5pyFile, fixture, Create temporary HDF5 files for testing.

### Community 349 - "TestNoPySide6DirectImports"
Cohesion: 0.18
Nodes (7): fixture, Test that source files don't have direct PySide6 imports., Get the project root directory., T056: Verify no direct PySide6 imports in gui/ (except qt_compat)., T056: Verify no direct PySide6 imports in module/ and threading/., T056: Verify only auto-generated files have direct PySide6 imports., TestNoPySide6DirectImports

### Community 350 - "test_package_basics.py"
Cohesion: 0.24
Nodes (10): unit, Unit tests for basic package functionality. This module provides unit tests for…, Test that package version is accessible., Test that CLI module can be imported., Test that basic modules can be imported., Test that threading components can be imported without metaclass conflicts., test_basic_imports(), test_cli_module_importable() (+2 more)

### Community 351 - "MockQtEnvironment"
Cohesion: 0.22
Nodes (7): mock_qt_environment(), MockQtEnvironment, Mock Qt environment for isolated testing., Set up isolated Qt environment for testing., Create mock timer for testing threading violations., Clean up mock environment., Fixture for mock Qt environment.

### Community 352 - "TestPerformanceMonitor"
Cohesion: 0.22
Nodes (6): Monitor test performance and detect slow/flaky tests., Record execution time for a test., Get list of tests that exceed the slow threshold., Get list of tests with high variance in execution times., Generate comprehensive performance report., TestPerformanceMonitor

### Community 353 - "take_snapshot"
Cohesion: 0.33
Nodes (8): Namespace, configure_offscreen_env(), main(), parse_args(), Path, Force deterministic, headless Qt settings., Launch the viewer offscreen and capture a snapshot to the given path., take_snapshot()

### Community 354 - "._connect_signals"
Cohesion: 0.20
Nodes (6): Any, Initialize SimpleMask window. Args: parent_viewer: Reference to parent XPCS…, Connect UI signals to handlers., Handle drawing tool selection. Args: tool_key: Key of selected tool from…, Handle eraser tool selection., Synchronize toolbar action checked states. Args: selected_tool: Currently…

### Community 355 - ".get_health_summary"
Cohesion: 0.22
Nodes (8): get_health_status(), Any, Track object for health monitoring., Get comprehensive health summary., Get health improvement recommendations., Get current health status summary., Track object for health monitoring., track_object_health()

### Community 356 - "HealthMonitor"
Cohesion: 0.20
Nodes (6): HealthMonitor, Initialize core health metrics with appropriate thresholds., Register a callable to be invoked once per monitoring interval. This allows…, Remove a previously registered periodic callback., Get monitoring performance impact statistics., Non-intrusive health monitoring system using background threads. Monitors…

### Community 357 - "TestModuleIntegration"
Cohesion: 0.20
Nodes (6): Test suite for module-level integration., Test that logger is properly initialized., Test that PyQtGraph configuration is set., Test that required modules are imported., Test that module constants are accessible., TestModuleIntegration

### Community 358 - "TestPartitionMemoryEfficiency"
Cohesion: 0.20
Nodes (6): Memory efficiency tests for partition operations., Verify linear partition produces correct results., Verify logarithmic partition produces correct results., Record timing for linear partition., Record timing for logarithmic partition., TestPartitionMemoryEfficiency

### Community 359 - "TestFileLoading"
Cohesion: 0.20
Nodes (6): Test suite for file loading and validation., Test loading a single HDF5 file., Test loading multiple files simultaneously., Test file validation status display., Test handling of invalid or corrupted files., TestFileLoading

### Community 360 - "fixture"
Cohesion: 0.15
Nodes (13): capture_logs(), fixtures_dir(), performance_timer(), fixture, Logger, Path, Create temporary file path., Path to test fixtures directory. (+5 more)

### Community 361 - "TestBackendArrayCreation"
Cohesion: 0.20
Nodes (5): Test array creation methods work correctly., Test arange creation., Test linspace creation., Test meshgrid creation., TestBackendArrayCreation

### Community 362 - "TestSpecificFittingModels"
Cohesion: 0.20
Nodes (6): Test specific fitting models used in XPCS analysis, Test G2 single exponential fitting model, Test G2 double exponential fitting model, Test SAXS form factor fitting (sphere model), Test power law fitting with physical constraints, TestSpecificFittingModels

### Community 363 - "TestViewerKernelAverageWorker"
Cohesion: 0.20
Nodes (6): Test suite for ViewerKernel single average worker., Test average worker is None on init., Test remove_job sets worker to None., Test update_avg_info is safe when no worker exists., Test update_avg_info calls worker.update_plot()., TestViewerKernelAverageWorker

### Community 364 - "TestAPS8IDIKeyStructure"
Cohesion: 0.20
Nodes (6): Test suite for APS 8IDI key structure validation., Test that key is a dictionary., Test that 'nexus' key exists in main key dict., Test that all essential keys are present., Test that two-time correlation keys exist., TestAPS8IDIKeyStructure

### Community 365 - "TestAPS8IDISpecificPaths"
Cohesion: 0.20
Nodes (6): Test suite for specific APS 8IDI path validation., Test specific qmap path values., Test specific analysis path values., Test detector-specific path values., Test beam-specific path values., TestAPS8IDISpecificPaths

### Community 366 - "TestBayesianIntegration"
Cohesion: 0.20
Nodes (6): End-to-end test of the batch Bayesian pipeline., Setting bayesian_fit_summary must not affect fit_summary on XpcsFile., assemble_fit_summary output must have source='bayesian'., plot_bayesian_all_q must return a Figure., plot_bayesian_all_q must work when data_t_el differs from summary t_el., TestBayesianIntegration

### Community 367 - "safe_version"
Cohesion: 0.20
Nodes (6): Verify safe_version returns a string., Verify safe_version returns 'unknown' for non-existent packages., Verify safe_version never raises exceptions., Safely retrieve package version for reproducibility tracking. Per Technical…, Convert to serializable dictionary. Per Technical Guidelines, exports include:…, safe_version()

### Community 369 - "test_tg6_fitting_p1.py"
Cohesion: 0.03
Nodes (61): Array, skipif, Tests for NumPyro model definitions (T034). This module tests the NumPyro…, Test model function can be imported., Test model includes two relaxation times., Tests for power law model., Test model function can be imported., Test model includes power law exponent alpha. (+53 more)

### Community 370 - "get_health_monitor"
Cohesion: 0.15
Nodes (9): QCloseEvent, get_health_monitor(), health_monitoring_context, Start background health monitoring., Get or create the global health monitor instance., Start background health monitoring., Context manager for automatic health monitoring during operations., start_health_monitoring() (+1 more)

### Community 371 - "run_comprehensive_validation"
Cohesion: 0.25
Nodes (8): generate_validation_report, run_comprehensive_validation, validate_correlation_functions, validate_fitting_algorithms, validate_fourier_transforms, validate_intensity_normalization, validate_q_space_calculations, validate_statistical_properties

### Community 372 - "TestMemoryAndResourceErrors"
Cohesion: 0.25
Nodes (5): Test suite for memory and resource limitation scenarios., Test handling of memory errors with large datasets., Test handling of disk space errors during operations., Test behavior when thread pool is exhausted., TestMemoryAndResourceErrors

### Community 373 - "TestCommandPaletteExecution"
Cohesion: 0.20
Nodes (6): Tests for action execution., Executing action should call callback., Executing action should emit action_triggered signal., Disabled actions should not appear in results., Enabled actions should appear in results., TestCommandPaletteExecution

### Community 375 - "ScientificValidationFramework"
Cohesion: 0.25
Nodes (6): Calculate G2 correlation function (mock implementation)., Generate signal with exponential correlation., Generate realistic speckle intensity signal., Framework for validating scientific accuracy of XPCS algorithms., Validate correlation function calculations against theoretical results., ScientificValidationFramework

### Community 376 - "TestDragDropListViewInit"
Cohesion: 0.20
Nodes (6): Tests for DragDropListView initialization., DragDropListView should initialize with drag-drop enabled., is_drag_enabled should return True by default., set_drag_enabled(False) should disable drag-drop., set_drag_enabled(True) should enable drag-drop., TestDragDropListViewInit

### Community 377 - "test_recent_paths.py"
Cohesion: 0.24
Nodes (7): Unit tests for RecentPathsManager., Tests for get_recent_paths_file function., get_recent_paths_file should return a Path object., Recent paths file should be in .xpcsviewer directory., TestGetRecentPathsFile, get_recent_paths_file(), Get the path to the recent paths file.

### Community 378 - "RecentPath"
Cohesion: 0.24
Nodes (7): Tests for RecentPath dataclass., RecentPath should store all fields., RecentPath should default to access_count=1., TestRecentPath, Get recent paths ordered by last access time. Returns: List of RecentPath…, A recently accessed directory., RecentPath

### Community 379 - "TestToastStyling"
Cohesion: 0.20
Nodes (6): Tests for toast CSS styling., INFO toast should have correct objectName for CSS., SUCCESS toast should have correct objectName for CSS., WARNING toast should have correct objectName for CSS., ERROR toast should have correct objectName for CSS., TestToastStyling

### Community 380 - "TestGetToolColor"
Cohesion: 0.20
Nodes (6): Tests for get_tool_color function., get_tool_color should return red for exclusive., get_tool_color should return green for inclusive., get_tool_color should return red for unknown mode., get_tool_color should return red for empty mode., TestGetToolColor

### Community 381 - "TestDefaultDrawParams"
Cohesion: 0.20
Nodes (6): Tests for default drawing parameters., DEFAULT_DRAW_PARAMS should be defined., Default radius should be positive., Default width should be positive., Default movable should be True., TestDefaultDrawParams

### Community 382 - "TestWindowPartitionControls"
Cohesion: 0.20
Nodes (6): Tests for partition controls in SimpleMask window., Window should have partition parameter spinboxes., Window should have Compute Partition button., Window should have Show Partition toggle button., Window should have Export to Viewer button., TestWindowPartitionControls

### Community 383 - "ensure_numpy"
Cohesion: 0.02
Nodes (141): G2 Analysis Algorithm Validation Tests This module provides comprehensive…, Test G2 interpolation accuracy and preservation of properties, Test that interpolation preserves G2 mathematical properties, Test interpolation accuracy against analytical solution, TestG2InterpolationAccuracy, memory_test_data(), fixture, Benchmark tests for memory-efficient operations. Verifies memory reduction from… (+133 more)

### Community 384 - "QtErrorCapture"
Cohesion: 0.14
Nodes (10): QtErrorCapture, Check if any timer-related errors were captured., Check if any signal/slot connection errors were captured., Get summary of all captured errors., Capture and analyze Qt error messages., Test framework for Qt error regression testing., Qt message handler to capture Qt warnings and errors., Establish baseline for Qt error detection. (+2 more)

### Community 385 - "fixture"
Cohesion: 0.22
Nodes (5): make_g2_ensemble_data(), make_g2_fitting_data(), fixture, Single-exponential G2 model., _single_exp_model()

### Community 386 - "ScientificValidationTestSuite"
Cohesion: 0.15
Nodes (7): Unit test suite for scientific validation framework., Test that the validation framework initializes correctly., Test correlation function validation methods., Test statistical property validation methods., Test Fourier transform validation methods., Test that comprehensive validation runs without errors., ScientificValidationTestSuite

### Community 387 - "gui"
Cohesion: 0.28
Nodes (6): gui, Test recent directory tracking functionality., Test suite for directory selection and browsing functionality., Test directory selection dialog functionality., Test directory path display in GUI., TestDirectorySelection

### Community 388 - "TestSimpleMaskUnsavedChanges"
Cohesion: 0.25
Nodes (5): Tests for unsaved changes tracking., New window should have no unsaved changes., Marking unsaved should update state and title., Marking saved should clear unsaved state., TestSimpleMaskUnsavedChanges

### Community 389 - "2.1 HDF5 File I/O"
Cohesion: 0.33
Nodes (6): 2.1 HDF5 File I/O, 2. External Service Integrations, HDF5 Schema Contracts (Implicit), Integration Points (12 identified), ⚠️ Integration Risks, 🔧 Recommended Facade Pattern

### Community 390 - "3.1 Core Data Structures"
Cohesion: 0.33
Nodes (6): 3.1 Core Data Structures, 3. Shared Data Schemas, Backend Array Protocol, G2 Data Structure, Geometry Metadata (Cross-Module Contract), Q-Map Dictionary (Cross-Module Contract)

### Community 391 - "._save"
Cohesion: 0.25
Nodes (5): Path, Save recent paths to disk., Add or update a path in the recent list. Args: path: Directory path to add, Remove a path that no longer exists. Args: path: Path to remove Returns: True…, Clear all recent paths.

### Community 392 - ".__init__"
Cohesion: 0.25
Nodes (5): QWidget, Initialize the ToastManager. Args: parent: Parent window for toast positioning, Initialize a toast widget. Args: message: Text to display toast_type: Type…, Set up the widget layout., Apply styling based on toast type.

### Community 393 - ".request_cancel"
Cohesion: 0.25
Nodes (5): Slot, Request cancellation of the operation., Handle operation cancellation in the UI., Request cancellation of an operation (to be connected to actual cancellation)., Request cancellation of all operations.

### Community 394 - "ReliabilityContext"
Cohesion: 0.22
Nodes (5): Context manager for enhanced reliability with retries and exponential backoff.…, Execute ``func(*args, **kwargs)`` with automatic retry on failure. Retries up…, Context manager for enhanced reliability with retries and exponential backoff.…, reliability_context(), ReliabilityContext

### Community 395 - ".get_module"
Cohesion: 0.25
Nodes (4): ModuleType, Generate G2 correlation function plots for multi-tau analysis. Creates…, Generate G2 stability plots showing frame-by-frame correlation analysis.…, Public interface to lazy load analysis modules

### Community 396 - "TestG2ModConstants"
Cohesion: 0.25
Nodes (5): Test suite for G2 module constants., Test colors tuple is properly defined., Test symbols list is properly defined., Test specific color values match expected matplotlib colors., TestG2ModConstants

### Community 397 - "TestErrorHandling"
Cohesion: 0.25
Nodes (5): Test suite for error handling in G2 module., Test get_data behavior when XF object raises exception., Test compute_geometry with empty g2 list., Test compute_geometry with invalid g2 shape., TestErrorHandling

### Community 399 - "test_analysis_tabs.py"
Cohesion: 0.33
Nodes (4): Tests for analysis tab functionality and interactions. This module provides…, Test suite for diffusion analysis tab., Test diffusion tab initializes properly., TestDiffusionTab

### Community 400 - "TestG2AnalysisTab"
Cohesion: 0.25
Nodes (5): Test suite for G2 correlation analysis tab., Test G2 analysis tab initializes with proper components., Test G2 fitting parameter controls., Test G2 fitting execution via button clicks., TestG2AnalysisTab

### Community 401 - "TestUIBoundaryConditions"
Cohesion: 0.25
Nodes (5): Test suite for UI boundary conditions and edge cases., Test behavior with extreme window sizes., Test parameter controls at boundary values., Test handling of empty or minimal datasets., TestUIBoundaryConditions

### Community 402 - "TestEdgeCaseData"
Cohesion: 0.25
Nodes (5): Test suite for edge case data scenarios., Test handling of NaN and infinite values in data., Test handling of Unicode and special characters in metadata., Test handling of extremely large data arrays., TestEdgeCaseData

### Community 403 - "TestFileListManagement"
Cohesion: 0.25
Nodes (5): Test suite for file list display and management., Test file list widget display and population., Test file selection within file list widgets., Test right-click context menu on file items., TestFileListManagement

### Community 404 - "TestDragAndDrop"
Cohesion: 0.25
Nodes (5): Test suite for drag and drop file operations., Test drag and drop file loading., Test drag and drop of multiple files., Test drag and drop of invalid files., TestDragAndDrop

### Community 405 - "TestSimpleMaskDataLoading"
Cohesion: 0.25
Nodes (5): Tests for loading detector data into SimpleMask., Loading None should show empty canvas message., Loading valid detector image should display it., Loading data should update info label with image info., TestSimpleMaskDataLoading

### Community 406 - "StatisticalCrossValidator"
Cohesion: 0.29
Nodes (5): ndarray, Cross-validator using statistical methods like k-fold validation, Perform k-fold cross-validation on fitting algorithm Args: x_data: Independent…, Test statistical cross-validation framework, StatisticalCrossValidator

### Community 407 - "TestViewerKernelInit"
Cohesion: 0.25
Nodes (5): Test suite for ViewerKernel initialization., Test basic ViewerKernel initialization., Test ViewerKernel initialization with statusbar., Test that initialization calls reset_meta., TestViewerKernelInit

### Community 408 - "test_aps_8idi.py"
Cohesion: 0.29
Nodes (6): parametrize, Unit tests for APS 8IDI beamline-specific data structures. This module provides…, Test that keys in same category have consistent path prefixes., Test that critical keys exist in the key mapping., test_critical_keys_exist(), test_path_category_consistency()

### Community 409 - "TestAPS8IDICompatibility"
Cohesion: 0.25
Nodes (5): Test suite for APS 8IDI backward compatibility., Test that essential key paths haven't changed., Test that no two keys map to the same path., Test that paths follow NeXus standard conventions., TestAPS8IDICompatibility

### Community 410 - "TestAPS8IDIKeyValidation"
Cohesion: 0.25
Nodes (5): Test suite for APS 8IDI key validation., Test that key names are valid Python identifiers., Test that paths have reasonable depth., Test that paths contain required components., TestAPS8IDIKeyValidation

### Community 411 - "TestAPS8IDIDataTypes"
Cohesion: 0.25
Nodes (5): Test suite for APS 8IDI data type implications., Test that key structure behaves as expected for read-only access., Test that key mapping supports typical XPCS analysis workflow., Test that key mapping supports two-time correlation analysis., TestAPS8IDIDataTypes

### Community 412 - "TestBug022DoubleExpTauSorting"
Cohesion: 0.25
Nodes (5): BUG-022: tau1/tau2 must be sorted before computing tau2_factor., If NLSQ returns tau1 > tau2, sorting must ensure tau2_factor > 0., When tau1 < tau2 (normal case), sorting should not change anything., Verify the sampler source contains tau sorting logic (BUG-022)., TestBug022DoubleExpTauSorting

### Community 413 - "TestDragDropListViewWithModel"
Cohesion: 0.25
Nodes (5): Tests for DragDropListView with a model attached., get_item_order should return empty list for empty model., get_item_order should return indices in order., get_item_order should return empty list when no model., TestDragDropListViewWithModel

### Community 414 - "RecentPathsState"
Cohesion: 0.32
Nodes (6): Tests for RecentPathsState dataclass., RecentPathsState should have sensible defaults., RecentPathsState should accept custom values., TestRecentPathsState, Recent directories state., RecentPathsState

### Community 415 - "4. Compound Bugs — Issues That Must Be Fixed Together"
Cohesion: 0.33
Nodes (6): 4. Compound Bugs — Issues That Must Be Fixed Together, COMPOUND-A — Bayesian fit is completely broken (P0-01 + P0-03 + P1-18), COMPOUND-B — Schema dtype validation rejects all legacy HDF5 data (P0-04 + P0-05 + P1-15), COMPOUND-C — `QMapSchema` access broken at two levels (P0-02 + type-analyzer note), COMPOUND-D — Shutdown race: Bayesian worker flag left set + UI buttons disabled (P1-04 + P1-14), COMPOUND-E — Stale plot on cancel is both a threading issue and a state issue (P1-01 + P1-09)

### Community 416 - "6. Architecture Improvement Recommendations"
Cohesion: 0.33
Nodes (6): 6. Architecture Improvement Recommendations, A. Centralize schema coercion in `HDF5Facade`, B. Make `BackendProtocol` enforceable, C. Decouple `NLSQResult` legacy fields from `native_result` delegation, D. Move `safe_shutdown()` and `closeEvent` shutdown logic to a single `ApplicationShutdown` class, E. Introduce a `DatasetContext` object to carry active-file state

### Community 417 - "test_shortcut_manager.py"
Cohesion: 0.25
Nodes (5): Unit tests for ShortcutManager., Tests for shortcut execution., Shortcut callback should be connected., shortcut_triggered signal should be emitted., TestShortcutExecution

### Community 418 - "comprehensive_xpcs_hdf5 (fixture)"
Cohesion: 0.40
Nodes (6): comprehensive_xpcs_hdf5 (fixture), detector_geometry (fixture), minimal_xpcs_hdf5 (fixture), qmap_data (fixture), synthetic_correlation_data (fixture), synthetic_scattering_data (fixture)

### Community 419 - "twotime_utils.py"
Cohesion: 0.11
Nodes (21): cleanup_shared_arrays(), correct_diagonal_c2_vectorized(), create_shared_array(), get_all_c2_from_hdf(), get_c2_stream(), get_shared_array(), get_single_c2_from_hdf(), ndarray (+13 more)

### Community 420 - "TestShortcutQuery"
Cohesion: 0.25
Nodes (5): Tests for shortcut query methods., get_registered_shortcuts should return list of IDs., is_registered should return True for registered shortcut., is_registered should return False for unregistered shortcut., TestShortcutQuery

### Community 421 - "TestShortcutConflictDetection"
Cohesion: 0.25
Nodes (5): Tests for key sequence conflict detection., Registering the same key sequence under a different ID should fail., Re-registering the same ID (even same key) should fail., After unregistering, the key sequence should be available again., TestShortcutConflictDetection

### Community 422 - ".get_memory_usage"
Cohesion: 0.05
Nodes (38): svmem, get_file_info(), Get basic file information and statistics. Parameters ---------- fname : str…, get_memory_predictor(), MemoryPrediction, OperationProfile, predict_operation_memory(), Any (+30 more)

### Community 423 - "test_drawing_tools.py"
Cohesion: 0.25
Nodes (5): Unit tests for SimpleMask drawing tools configuration. Tests the drawing tool…, Tests that tool types match expected ROI types., All drawing tools should have valid tool types., Eraser should have valid tool type., TestToolTypeValidity

### Community 424 - "TestToolColors"
Cohesion: 0.25
Nodes (5): Tests for tool color configuration., TOOL_COLORS should be defined., Exclusive mode should have red color., Inclusive mode should have green color., TestToolColors

### Community 425 - "TestQMapCacheNoCopy"
Cohesion: 0.25
Nodes (5): Two calls with identical params return the same cached dict object., Test 3: Q-map cache hit returns the cached dict directly, no deep copy., compute_transmission_qmap must not call copy.deepcopy on cache hit., compute_reflection_qmap must not call copy.deepcopy on cache hit., TestQMapCacheNoCopy

### Community 426 - "XPCS logo 128x128"
Cohesion: 0.40
Nodes (5): XPCS logo 128x128, XPCS logo 16x16, XPCS logo 32x32, XPCS logo 512x512, XPCS logo 64x64

### Community 427 - "get_qmap"
Cohesion: 0.33
Nodes (4): MockH5py, Edge case and boundary condition tests. This module tests boundary conditions,…, get_qmap(), test_qmap_manager()

### Community 428 - "TestHealthMonitorGCDelta"
Cohesion: 0.25
Nodes (5): Test 5: GC metrics use per-interval delta, not stale start time., _update_gc_metrics must compute delta against last-interval stats, not start., GC metric update computes delta relative to previous interval's stats., _get_hdf5_connection_count must actually try to query pool size (BUG-053)., TestHealthMonitorGCDelta

### Community 429 - "TestNonzeroNoRecompilation"
Cohesion: 0.25
Nodes (5): Test 2: nonzero() in _jax_backend.py prevents per-input-size recompilation., nonzero() without explicit size uses x.size to keep shape fixed., nonzero() with explicit size pads results to fixed length., nonzero() without size defaults to x.size for JIT stability., TestNonzeroNoRecompilation

### Community 430 - "ThreadSafeQtDecorator"
Cohesion: 0.25
Nodes (5): Decorators for enforcing Qt thread safety., Decorator to ensure function runs in main Qt thread., Decorator to validate QTimer creation., Decorator to monitor function for threading violations., ThreadSafeQtDecorator

### Community 431 - "TestStabilizer"
Cohesion: 0.25
Nodes (5): Timeout decorator for tests that might hang., Decorator to make timing-dependent tests more deterministic., Stabilize flaky tests through various techniques., Retry decorator for flaky tests., TestStabilizer

### Community 432 - "scientific_validation.py"
Cohesion: 0.40
Nodes (4): BenchmarkData, convert_value(), Benchmark dataset for validation., Convert numpy types to JSON-serializable types.

### Community 433 - "._populate_results"
Cohesion: 0.25
Nodes (4): Show the command palette dialog., Filter results based on search query., Populate the results list with matching actions., Check if query fuzzy-matches text. Args: query: Search query (lowercase) text:…

### Community 434 - ".saxs_2d"
Cohesion: 0.20
Nodes (7): setter, Access SAXS 2D data with transparent lazy loading support. Returns either…, Set SAXS 2D data (regular array or lazy proxy)., Access SAXS 2D log data with transparent lazy loading support., Set SAXS 2D log data (regular array or lazy proxy)., Backward compatibility property for SAXS 2D data access. This property ensures…, Set SAXS 2D data (backward compatibility).

### Community 435 - "10. Conclusion"
Cohesion: 0.50
Nodes (4): 10. Conclusion, Priority Actions, Strengths, Weaknesses

### Community 436 - "1. Internal Module Dependencies"
Cohesion: 0.50
Nodes (4): 1.1 Dependency Graph, 1.2 High Fan-In Modules (Integration Hotspots), 1.3 High Fan-Out Modules (Brittle Dependencies), 1. Internal Module Dependencies

### Community 437 - "4. Cross-Module Data Flows"
Cohesion: 0.50
Nodes (4): 4.1 Primary Data Flow: XPCS Analysis Pipeline, 4.2 SimpleMask Data Flow, 4.3 Fitting Data Flow, 4. Cross-Module Data Flows

### Community 438 - "6. Recommended Architecture Patterns"
Cohesion: 0.50
Nodes (4): 6.1 Adapter Pattern for Backend I/O, 6.2 Repository Pattern for HDF5 Access, 6.3 Event-Driven Integration (Already Implemented ✅), 6. Recommended Architecture Patterns

### Community 439 - "7. Migration Roadmap"
Cohesion: 0.50
Nodes (4): 7. Migration Roadmap, Phase 1: Non-Breaking Additions (Weeks 1-4), Phase 2: Gradual Migration (Weeks 5-12), Phase 3: Cleanup and Optimization (Weeks 13-16)

### Community 440 - "9. Performance Implications"
Cohesion: 0.50
Nodes (4): 9. Performance Implications, Backend Conversion Overhead, Connection Pooling Impact, JIT Compilation Benefits

### Community 441 - "Appendix A: Data Structure Reference"
Cohesion: 0.50
Nodes (4): A.1 Complete QMapDict Schema, A.2 Complete GeometryMetadata Schema, A.3 Partition Dictionary Schema, Appendix A: Data Structure Reference

### Community 442 - ".setup_ui"
Cohesion: 0.29
Nodes (4): OperationInfo, QWidget, Set up the progress dialog UI., Information about a running operation.

### Community 443 - "test_parametrized_invalid_data_types"
Cohesion: 0.29
Nodes (7): object, parametrize, Parametrized test for various invalid data types., test_parametrized_invalid_data_types(), loadable_files(), fixture, Make ``add_target(preload=True)`` accept any existing file. Empty ``.h5`` files…

### Community 445 - "TestMatplotlibIntegration"
Cohesion: 0.20
Nodes (7): FigureCanvasQTAgg, fixture, Test suite for Matplotlib canvas integration., Create a Matplotlib canvas for testing., Test Matplotlib canvas creation and basic plotting., Test Matplotlib canvas mouse interactions., TestMatplotlibIntegration

### Community 446 - "test_file_operations.py"
Cohesion: 0.29
Nodes (5): MockH5py, Tests for file operations and data loading GUI functionality. This module tests…, Integration tests for file operations with other GUI components., Test that file loading updates analysis tabs., TestFileOperationIntegration

### Community 447 - "TestCommandPaletteKeyboard"
Cohesion: 0.29
Nodes (5): fixture, Tests for keyboard navigation., Create palette with test actions., Escape key should hide palette., TestCommandPaletteKeyboard

### Community 448 - ".window_and_manager"
Cohesion: 0.29
Nodes (4): fixture, Create manager with shortcuts for testing., Create window and manager for testing., Create window and manager for testing.

### Community 449 - ".__init__"
Cohesion: 0.29
Nodes (4): QWidget, Initialize the CommandPalette. Args: parent: Parent widget, Set up the dialog UI., Set up keyboard navigation.

### Community 450 - ".start_operation"
Cohesion: 0.29
Nodes (4): log_timing, Add a new operation to track., Start tracking a new operation. Args: operation_id: Unique identifier for the…, Show progress dialog if operation is still running.

### Community 451 - "TestDiagonalCorrectionPerformance"
Cohesion: 0.33
Nodes (4): Record timing for vectorized diagonal correction., Benchmark tests for diagonal correction., Verify vectorized diagonal correction produces correct results., TestDiagonalCorrectionPerformance

### Community 452 - "TestDataGenerator"
Cohesion: 0.33
Nodes (4): Generators for creating synthetic test data with controlled properties., Generate synthetic G2 correlation function data. Args: tau_range: Range of tau…, Generate synthetic SAXS scattering data. Args: q_range: Range of Q values (min,…, TestDataGenerator

### Community 453 - "TestDebugger"
Cohesion: 0.33
Nodes (4): Tools for debugging test failures and analyzing test behavior., Decorator to capture test context on failure. Usage:…, Decorator to log test execution steps. Usage: @TestDebugger.log_test_steps def…, TestDebugger

### Community 454 - "TestSAXS1DTab"
Cohesion: 0.33
Nodes (4): Test SAXS 1D tab initializes with proper components., Test SAXS 1D plot scaling and axis controls., Test suite for SAXS 1D analysis tab., TestSAXS1DTab

### Community 455 - "TestTwoTimeTab"
Cohesion: 0.33
Nodes (4): Test suite for two-time correlation analysis tab., Test two-time correlation tab initializes properly., Test two-time correlation parameter controls., TestTwoTimeTab

### Community 456 - "TestStabilityTab"
Cohesion: 0.33
Nodes (4): Test suite for stability analysis tab., Test stability analysis tab initializes properly., Test stability plot updates with mock data., TestStabilityTab

### Community 457 - "qt_application (fixture)"
Cohesion: 0.50
Nodes (4): qt_application (fixture), qt_cleanup (autouse fixture), qt_main_window (fixture), qt_widget (fixture)

### Community 458 - "TestMetadataTab"
Cohesion: 0.33
Nodes (4): Test suite for metadata display tab., Test metadata tab initializes properly., Test metadata display functionality., TestMetadataTab

### Community 459 - "TestTabIntegration"
Cohesion: 0.33
Nodes (4): Test suite for cross-tab interactions and data consistency., Test that data remains consistent across tab switches., Test that tab states are preserved during navigation., TestTabIntegration

### Community 460 - "TestSignalSlotErrors"
Cohesion: 0.33
Nodes (4): Test suite for signal/slot connection errors., Test behavior when signals are disconnected or fail., Test handling of exceptions in slot functions., TestSignalSlotErrors

### Community 461 - "TestViewerKernelProperties"
Cohesion: 0.33
Nodes (4): Test suite for ViewerKernel properties and attributes., Test path property access., Test statusbar property access., TestViewerKernelProperties

### Community 462 - "TestProgressIndication"
Cohesion: 0.33
Nodes (4): Test suite for progress indication during file operations., Test progress bar appears during loading operations., Test loading indicators during file operations., TestProgressIndication

### Community 463 - "TestQtCompatWithPyQt6"
Cohesion: 0.33
Nodes (4): Test qt_compat with PyQt6 backend (T058). Uses mocking to verify qt_compat…, T058: Verify qt_compat sets QT_API and delegates to qtpy for PyQt6., T058: Verify qtpy would route to PyQt6 when QT_API=pyqt6., TestQtCompatWithPyQt6

### Community 464 - "TestMaskExportContent"
Cohesion: 0.33
Nodes (4): Tests for mask content during export., Exported mask should be boolean array., Default mask (no edits) should be all True., TestMaskExportContent

### Community 466 - "TestPartitionSignalExport"
Cohesion: 0.33
Nodes (4): Tests for qmap_exported signal functionality., qmap_exported signal should exist on window., export_partition_to_viewer should emit partition dict., TestPartitionSignalExport

### Community 467 - "TestViewerKernelPerformance"
Cohesion: 0.33
Nodes (4): Test suite for ViewerKernel performance characteristics., Test ViewerKernel initialization performance., Test meta reset performance., TestViewerKernelPerformance

### Community 468 - "TestAsyncG2ResultHandling"
Cohesion: 0.33
Nodes (4): apply_g2_result must consume figure or axes handles from result dict., BUG-014: apply_g2_result() must use the pre-computed worker result directly and…, apply_g2_result must consume result dict without calling vk.plot_g2., TestAsyncG2ResultHandling

### Community 469 - "TestAPS8IDIKeyAccess"
Cohesion: 0.33
Nodes (4): Test suite for APS 8IDI key access patterns., Test accessing keys by category., Test that key lookup works as expected., TestAPS8IDIKeyAccess

### Community 470 - "BayesianDiagnosisWindow"
Cohesion: 0.07
Nodes (21): BayesianDiagnosisWindow, ConvergenceSummaryWidget, Any, QMainWindow, QWidget, Bayesian diagnosis window for MCMC fit results. Provides a shared diagnostic…, Diagnostic window showing Bayesian fit results and MCMC diagnostics. Reused for…, Set axis labels for the posterior predictive plot. (+13 more)

### Community 471 - "System Architecture Overview"
Cohesion: 0.67
Nodes (3): XPCS Data Flow Pipeline, Module Dependency Analysis, System Architecture Overview

### Community 472 - "TestQMapUtilityMethods"
Cohesion: 0.33
Nodes (4): Test suite for QMap utility methods., Test _get_default_value method., Test _create_minimal_fallback method., TestQMapUtilityMethods

### Community 473 - "TestQMapEdgeCases"
Cohesion: 0.33
Nodes (4): Test suite for QMap edge cases., Test QMap with None filename., Test QMap with empty root key., TestQMapEdgeCases

### Community 474 - "TestXpcsFileAttributeCollision"
Cohesion: 0.33
Nodes (4): BUG-015: XpcsFile.__dict__.update() must not silently overwrite existing…, The collision guard logic must prevent 'qmap'/'label' from being overwritten., XpcsFile.__init__ source must contain a collision guard., TestXpcsFileAttributeCollision

### Community 476 - "calibration.py"
Cohesion: 0.21
Nodes (14): minimize_with_grad(), Minimize objective function using gradient descent. Simple gradient descent…, compute_center_from_ring(), create_calibration_objective(), minimize_with_grad(), bool_, floating, NDArray (+6 more)

### Community 477 - "test_drag_drop_list.py"
Cohesion: 0.33
Nodes (4): Unit tests for DragDropListView widget., Tests for selection mode., Default selection mode should be SingleSelection., TestDragDropListViewSelection

### Community 478 - "DragDropListView"
Cohesion: 0.13
Nodes (9): DropAction, DragDropListView, Get current item order as indices. Returns: List of original indices in current…, Programmatically move an item from one position to another. This is useful for…, QListView subclass with internal drag-and-drop reordering. Emits…, Enable or disable drag-and-drop. Args: enabled: Whether drag-drop is enabled, Check if drag-and-drop is enabled. Returns: True if enabled, Override to track the starting index of a drag operation. Args:… (+1 more)

### Community 479 - "TestDragDropListViewSignals"
Cohesion: 0.33
Nodes (4): Tests for signal emission., DragDropListView should have items_reordered signal., items_reordered signal should be connectable., TestDragDropListViewSignals

### Community 480 - "TestRecentPathsManagerGetPaths"
Cohesion: 0.33
Nodes (4): Tests for RecentPathsManager.get_recent_paths., get_recent_paths should return empty list initially., get_recent_paths should return a copy., TestRecentPathsManagerGetPaths

### Community 481 - "TestRecentPathsManagerRemoveInvalid"
Cohesion: 0.33
Nodes (4): Tests for RecentPathsManager.remove_invalid_path., remove_invalid_path should remove the path., remove_invalid_path should return False for unknown path., TestRecentPathsManagerRemoveInvalid

### Community 482 - "TestNoScipyInterpolateImports"
Cohesion: 0.33
Nodes (4): T040: Verify no scipy.interpolate imports in module/ directory., Verify grep finds no scipy.interpolate imports in module/., Verify grep finds no direct scipy.ndimage imports in module/., TestNoScipyInterpolateImports

### Community 483 - "TestG2Interpolation"
Cohesion: 0.33
Nodes (4): T041: Test vectorized_g2_interpolation produces correct output., Verify G2 interpolation produces reasonable output., Verify G2 interpolation output shape is correct., TestG2Interpolation

### Community 484 - "TestQmapOverlay"
Cohesion: 0.33
Nodes (4): Tests for Q-map overlay display., Window should track Q-map overlay state., Q-map toggle should require loaded data., TestQmapOverlay

### Community 485 - "ProgressIndicator"
Cohesion: 0.40
Nodes (4): ProgressIndicator, Update the progress indicator., Calculate and update estimated time to completion., Individual progress indicator widget for a single operation.

### Community 486 - "TestIntegratedQtErrorScenarios"
Cohesion: 0.40
Nodes (4): slow, Test integrated Qt error scenarios., Test Qt errors in XPCS viewer context., TestIntegratedQtErrorScenarios

### Community 487 - ".from_numpy"
Cohesion: 0.40
Nodes (3): ndarray, Convert array to NumPy ndarray., Convert NumPy ndarray to backend array.

### Community 488 - "improve_control_panel_layout"
Cohesion: 0.50
Nodes (5): apply_group_box_styling(), improve_control_panel_layout(), QGroupBox, Improve the layout of a control panel group box. Args: group_box: The control…, Apply semantic styling to a QGroupBox. Args: group_box: The group box to style…

### Community 489 - "TestQMapTab"
Cohesion: 0.50
Nodes (3): Test suite for Q-map analysis tab., Test Q-map tab initializes properly., TestQMapTab

### Community 490 - "TestAverageTab"
Cohesion: 0.50
Nodes (3): Test suite for averaging functionality tab., Test averaging tab initializes properly., TestAverageTab

### Community 491 - "verify_diffusion_constraints"
Cohesion: 0.50
Nodes (4): Verify power law scaling relationships: y ∝ x^α Args: x_values: Independent…, Verify physical constraints for diffusion analysis Args: tau_values: Relaxation…, verify_diffusion_constraints(), verify_power_law_scaling()

### Community 492 - "TestQMapCaching"
Cohesion: 0.50
Nodes (3): Test suite for QMap caching mechanisms., Test that caching structures are properly initialized., TestQMapCaching

### Community 493 - "TestRecentPathsManagerClear"
Cohesion: 0.50
Nodes (3): Tests for RecentPathsManager.clear., clear should remove all paths., TestRecentPathsManagerClear

### Community 494 - "Multi-Layered Test Framework"
Cohesion: 0.67
Nodes (3): Error Handling Test Suite, GUI Interactive Testing (pytest-qt), Multi-Layered Test Framework

### Community 498 - "twotime_batch.py"
Cohesion: 0.12
Nodes (28): XpcsFile, create_twotime_plot_matplotlib(), extract_q_phi_from_label(), find_hdf_files(), find_qbin_for_qphi(), find_qbins_for_phi(), find_qbins_for_q(), generate_output_filename() (+20 more)

### Community 502 - ".stop_monitoring"
Cohesion: 0.50
Nodes (3): Stop background health monitoring. The lock is released before joining the…, Stop background health monitoring., stop_health_monitoring()

### Community 505 - "create_separator"
Cohesion: 0.67
Nodes (3): create_separator(), QFrame, Create a visual separator line. Args: orientation: Line direction Returns:…

### Community 543 - "AverageToolbox"
Cohesion: 0.06
Nodes (23): Unit tests for BUG-009: threading.Event cross-thread cancellation in…, Calling kill() multiple times must not raise and event stays set., Two AverageToolbox instances must have independent is_killed events., Test 1: is_killed as threading.Event provides proper cross-thread cancellation., is_killed must be a threading.Event, not a plain bool., is_killed must start in the unset (not killed) state., Calling kill() must set the threading.Event., threading.Event guarantees that set() is visible to all threads.… (+15 more)

### Community 595 - "Interp1d"
Cohesion: 0.18
Nodes (13): JAX replacements for SciPy functions used in the SimpleMask module. Provides…, Interp1d, interp2d_jax(), ArrayLike, ndarray, JAX replacements for scipy.interpolate functions using interpax. This module…, Interpolation using interpax library., NumPy/SciPy fallback implementation. (+5 more)

### Community 602 - "test_g2mod.py"
Cohesion: 0.09
Nodes (22): parametrize, Unit tests for G2 analysis module. This module provides comprehensive unit…, Test suite for data processing helper functions., Test that get_data pre-allocates lists for memory efficiency., Test compute_geometry with different plot types., Test compute_geometry scaling with different data sizes., Test suite for performance characteristics., Test get_data performance with large file list. (+14 more)

### Community 607 - "vectorized_background_subtraction"
Cohesion: 0.22
Nodes (7): Test background subtraction with error propagation, T042: Test vectorized_background_subtraction produces correct output., Verify background subtraction with same q-values., Verify background subtraction with different q-values requiring interpolation., TestVectorizedBackgroundSubtraction, Vectorized background subtraction with error propagation. Args:…, vectorized_background_subtraction()

### Community 608 - "test_g2_saxs_opt.py"
Cohesion: 0.08
Nodes (39): area_norm_data(), _area_norm_loop(), batch_norm_data(), _batch_norm_loop(), binning_1d_data(), binning_2d_data(), benchmark, fixture (+31 more)

### Community 618 - "TestGPULaunch"
Cohesion: 0.11
Nodes (12): gpu, skipif, Tests for GPU system launch (T069). Tests that application launches correctly…, Tests for GPU system launch., Test backend detects GPU when available., Test GPU device is listed in available devices., Test Q-map computation runs on GPU., Tests for GPU detection logic. (+4 more)

### Community 621 - "test_nlsq_jit_tracing.py"
Cohesion: 0.10
Nodes (19): needs_nlsq, Regression tests for nlsq JIT-tracing compatibility. Verifies that model…, Model functions must survive nlsq JIT-tracing without…, single_exp_all must work with nlsq.curve_fit (JIT-traced)., double_exp_all must work with nlsq.curve_fit (JIT-traced)., power_law must work with nlsq.curve_fit (JIT-traced)., Model functions must use jnp.exp, not np.exp., Lint rule: xpcs_file/fitting.py must not use numpy math functions. Model… (+11 more)

### Community 628 - "test_batch_vectorize.py"
Cohesion: 0.17
Nodes (10): c2_batch_data(), fixture, Benchmark tests for vectorized batch processing. Verifies performance…, Benchmark tests for batch C2 operations., Verify batch C2 operations produce correct results., Generate C2 matrix batch data for benchmarks., Record timing for batch C2 operations., TestBatchC2Operations (+2 more)

### Community 629 - "TestC2StatisticsVectorized"
Cohesion: 0.22
Nodes (7): Benchmark tests for C2 statistics vectorization., Verify vectorized C2 statistics produces correct results., Record timing for vectorized C2 statistics., Verify off-diagonal mean matches loop-based calculation., TestC2StatisticsVectorized, compute_c2_statistics_vectorized(), Compute statistical measures for C2 matrices using vectorized operations. Uses…

### Community 720 - "TestCloseEventWaitsForThreadPool"
Cohesion: 0.33
Nodes (4): BUG-013: closeEvent must call thread_pool.waitForDone() to prevent signals…, closeEvent must invoke waitForDone on the thread pool., waitForDone call must include a timeout to avoid hanging., TestCloseEventWaitsForThreadPool

### Community 721 - "TestIntensityTimeTab"
Cohesion: 0.50
Nodes (3): Test suite for intensity vs time analysis tab., Test intensity-time tab initializes properly., TestIntensityTimeTab

## Knowledge Gaps
- **815 isolated node(s):** `rerun_baselines.sh script`, `JAX_PLATFORMS`, `MockH5py`, `MockH5py`, `MockH5py` (+810 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **257 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Structured Logging System` connect `xpcs_viewer.py` to `test_g2_partial_safety.py`, `nlsq_optimize`, `FitResult`, `SessionManager`, `BaseAsyncWorker`, `framework/utils.py`, `tests/conftest.py`, `Backend Abstraction Pattern`, `xpcsviewer/simplemask/__init__.py`, `XPCSBaseError`, `XPCS Viewer (xpcsviewer) Python Package`, `xpcsviewer/utils/reliability.py`, `get_backend`, `safe_json_write`, `ToastType`, `BayesianDiagnosisWindow`, `test_g2mod.py`, `calibration.py`, `export_bayesian_csv`, `ensure_numpy`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Why does `XpcsViewer` connect `XpcsViewer` to `QtErrorCapture`, `nlsq_optimize`, `ViewerKernel`, `QtThreadingValidator`, `SessionManager`, `xpcs_viewer.py`, `test_tab_availability.py`, `QMapSchema`, `MemoryTestUtils`, `get_icon`, `.update_tab_availability`, `test_tg3_mask_export_and_g2_plot.py`, `Ui_mainWindow`, `TestEnsureNumpyAtPyQtGraphBoundaries`, `TestInitAverageSaveNamePreservation`, `.capture_qt_warnings`, `test_qt_error_detection.py`, `MockH5pyFile`, `.get_selected_rows`, `TestSingletonDoubleCheckedLocking`, `TestCloseEventWaitsForThreadPool`, `QtTestRunner`, `TestAsyncG2ResultHandling`, `BayesianDiagnosisWindow`, `TestXpcsFileAttributeCollision`, `test_twotime_qbin_memory.py`, `gui`, `MockQtEnvironment`, `take_snapshot`, `TestIntegratedQtErrorScenarios`, `.load_path`, `TestQtTimerThreadingErrors`, `get_health_monitor`, `gui_interactive/conftest.py`, `.__init__`, `.on_async_plot_ready`, `ensure_numpy`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `Threading API Reference` connect `xpcs_viewer.py` to `AsyncViewerKernel`, `FitResult`, `xpcs_file.py`, `qt_threading_utils.py`, `BaseAsyncWorker`, `QMapSchema`, `get_memory_manager`, `AverageToolbox`, `state_validator.py`, `tests/utils/reliability.py`, `UnifiedMemoryManager`, `HDF5ConnectionPool`, `test_qt_error_detection.py`, `XPCS Viewer (xpcsviewer) Python Package`, `xpcsviewer/utils/reliability.py`, `isolation.py`, `MemoryMonitor`, `QtTestRunner`, `PooledConnection`, `ensure_numpy`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `XpcsViewer` (e.g. with `_QtErrorCapture` and `QtTestRunner`) actually correct?**
  _`XpcsViewer` has 45 INFERRED edges - model-reasoned connections that need verification._
- **Are the 68 inferred relationships involving `XpcsFile` (e.g. with `MockH5py` and `MockH5pyFile`) actually correct?**
  _`XpcsFile` has 68 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `FitResult` (e.g. with `TestChain1SilentScientificCorruption` and `TestChain2SignalChaosAtShutdown`) actually correct?**
  _`FitResult` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `ViewerKernel` (e.g. with `MockH5py` and `MockH5pyFile`) actually correct?**
  _`ViewerKernel` has 30 INFERRED edges - model-reasoned connections that need verification._