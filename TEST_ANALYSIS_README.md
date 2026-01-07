# XPCS Viewer - Legacy Component Test Coverage Analysis

**Date:** January 6, 2026
**Analysis Scope:** Test coverage for identified legacy/god-class modules
**Status:** Complete - Ready for Team Review

---

## Quick Summary

This analysis evaluates test coverage for 8 legacy/god-class modules in XPCS Viewer. Results show **critical gaps** in test coverage for core analysis functionality:

**Key Finding:** Two "god objects" (`XpcsFile` and `ViewerKernel`) combine 67 public methods with only ~5% actual test coverage despite having 89 test cases.

### Coverage at a Glance

```
XpcsFile (39 methods)        : 5%  tested  ❌ CRITICAL
ViewerKernel (28 methods)    : 4%  tested  ❌ CRITICAL
G2 Analysis (11 functions)   : 35% tested  ⚠️  MODERATE
TwoTime (4 functions)        : 30% tested  ⚠️  MODERATE
SAXS 1D (10 functions)       : 40% tested  ⚠️  MODERATE
FileIO (45+ methods)         : 20% tested  ⚠️  GOOD (infrastructure tested well)
────────────────────────────────────────────────────
Overall                      : 18% tested  ❌ CRITICAL
```

---

## Generated Documents

This analysis includes three detailed documents:

### 1. **TEST_COVERAGE_ANALYSIS.md** (24 KB)
**Purpose:** Comprehensive technical analysis with detailed findings

**Contents:**
- Executive summary with statistics
- Module-by-module analysis (8 modules covered)
- Public method inventory (80 items for XpcsFile, 28 for ViewerKernel)
- Critical untested paths (64 methods)
- Risk assessment (high/medium/low categories)
- Test architecture recommendations
- Coverage target roadmap (3-phase implementation plan)
- Detailed file location references

**Best For:** Architects, Test Leads, Team Planning

**Key Sections:**
- Risk Assessment (identifies CRITICAL paths like G2 fitting untested)
- Test Architecture (70/20/10 pyramid structure)
- Phase 1-3 Implementation Roadmap
- Metrics & Monitoring framework

**Read Time:** 30-40 minutes

---

### 2. **COVERAGE_SUMMARY.txt** (18 KB)
**Purpose:** Executive summary with visual formatting (quick reference)

**Contents:**
- One-page coverage table
- Critical untested methods list
- Risk areas (CRITICAL, HIGH, MEDIUM)
- Testing recommendations by phase
- Quick file location reference
- Next steps checklist

**Best For:** Managers, Quick Overview, Team Sync-ups

**Key Features:**
- Visual tables for easy scanning
- Color-coded priority levels
- At-a-glance metrics
- Ready-to-present format

**Read Time:** 10-15 minutes

---

### 3. **PRIORITY_TEST_SPECIFICATIONS.md** (19 KB)
**Purpose:** Actionable test specifications for Phase 1 implementation

**Contents:**
- 21 specific test cases (ready to implement)
- Test IDs and objectives
- Input/output specifications
- Assertion patterns
- Error case handling
- Test execution matrix
- Success criteria

**Best For:** Developers, QA Engineers, Sprint Planning

**Key Sections:**
- 4 test suites with specific test cases
- Mock/fixture recommendations
- Performance targets
- Execution time estimates (~40 hours for Phase 1)

**Read Time:** 20-30 minutes

---

## How to Use These Documents

### For Immediate Action (This Sprint)

1. **Start Here:** Read **COVERAGE_SUMMARY.txt** (10 min)
   - Understand the scope and priority
   - Review critical untested paths

2. **Detail Review:** Read **TEST_COVERAGE_ANALYSIS.md** sections:
   - Module you're responsible for (5-10 min each)
   - Risk Assessment section (5 min)

3. **Implementation:** Use **PRIORITY_TEST_SPECIFICATIONS.md**
   - Pick test cases XF-001 through VK-009
   - Follow exact specifications
   - Estimate: 2-3 tests per developer per day

### For Sprint Planning

1. Review **TEST_COVERAGE_ANALYSIS.md**: Coverage Target Roadmap
   - Phase 1: Critical Paths (60 hours, 16 tests)
   - Phase 2: Integration Tests (40 hours, 20 tests)
   - Phase 3: Edge Cases (50 hours, 35 tests)

2. Use **PRIORITY_TEST_SPECIFICATIONS.md**: Test Execution Matrix
   - Effort estimates per test (1-3 hours)
   - Dependencies between tests
   - Assign to team members

3. Set up coverage gates in CI/CD:
   - Minimum 70% for critical modules
   - Weekly trending reports
   - Block merge if below threshold

### For Architecture Review

1. Read **TEST_COVERAGE_ANALYSIS.md**:
   - God Object Pattern section
   - Architecture Debt section
   - Long-Term Recommendations

2. Consider refactoring legacy modules:
   - XpcsFile → DataManager + AnalysisEngine + ExportManager
   - ViewerKernel → PlottingOrchestrator + JobManager
   - Impact on test coverage reduction

---

## Key Findings Summary

### Critical Issues Found

**1. G2 Analysis Pipeline (Untested)**
- `get_g2_data()`, `fit_g2()` are untested
- Impact: Core XPCS feature unavailable for validation
- Risk Level: CRITICAL

**2. SAXS Visualization (Untested)**
- `plot_saxs_1d()`, `plot_saxs_2d()`, `add_roi()` untested
- Impact: All visualization features manually tested only
- Risk Level: CRITICAL

**3. ROI Extraction (Limited Testing)**
- `get_roi_data()`, `get_multiple_roi_data_parallel()` untested
- Impact: Custom analysis workflows not validated
- Risk Level: HIGH

**4. TwoTime Plotting (Untested)**
- `plot_twotime()`, `plot_twotime_g2()` untested
- Impact: TwoTime visualization not validated
- Risk Level: HIGH

### God Object Anti-pattern

**Identified:**
- `XpcsFile`: 39 public methods (data I/O + analysis + export + caching)
- `ViewerKernel`: 28 public methods (orchestration + 9 plots + 3 exports + jobs)

**Consequences:**
- Hard to test individual features
- Changes ripple across system
- High coupling makes refactoring difficult
- Test brittle and expensive to maintain

**Recommendation:** Decompose into focused components

---

## Metrics & Targets

### Coverage Goals (By Phase)

| Phase | XpcsFile | ViewerKernel | FileIO | Overall |
|-------|----------|--------------|--------|---------|
| Current | 5% | 4% | 20% | 18% |
| Phase 1 | 50% | 50% | 30% | 45% |
| Phase 2 | 70% | 70% | 50% | 65% |
| Phase 3 | 80% | 80% | 70% | 80% |

### Time Estimates

- **Phase 1 (Critical Paths):** 60 hours (16 tests)
- **Phase 2 (Integration):** 40 hours (20 tests)
- **Phase 3 (Edge Cases):** 50 hours (35 tests)
- **Total:** 150 hours over 3 sprints

---

## Implementation Checklist

### Immediate (This Week)

- [ ] Share analysis documents with team
- [ ] Schedule review meeting (30 min)
- [ ] Assign Phase 1 test implementation
- [ ] Set up test data fixtures
- [ ] Create CI/CD coverage gates

### Phase 1 (Next Sprint)

- [ ] Implement 16 priority tests (XF-001 through VK-009)
- [ ] Reach 50% coverage for XpcsFile, ViewerKernel
- [ ] Set up automated coverage reporting
- [ ] Conduct code review for new tests

### Phase 2 (Sprint +2)

- [ ] Implement 4 integration test suites
- [ ] Reach 70% coverage
- [ ] Add performance regression tests
- [ ] Document test patterns for team

### Phase 3 (Sprint +3)

- [ ] Implement 35 edge case tests
- [ ] Reach 80% coverage
- [ ] Add mutation testing
- [ ] Establish continuous monitoring

---

## Document File Locations

```
/Users/b80985/Projects/xpcsviewer/
├── TEST_COVERAGE_ANALYSIS.md          # Detailed technical analysis
├── COVERAGE_SUMMARY.txt                # Quick reference guide
├── PRIORITY_TEST_SPECIFICATIONS.md    # Implementation specs
└── TEST_ANALYSIS_README.md            # This file (navigation guide)
```

---

## Important Context

### What These Documents Cover

✓ Legacy/god-class modules analysis
✓ Public method inventory (80+ methods)
✓ Test coverage gaps (64 untested methods)
✓ Risk assessment and prioritization
✓ Actionable test specifications
✓ Implementation roadmap (3 phases)
✓ Architecture recommendations

### What These Documents Do NOT Cover

✗ Currently passing tests (only coverage analysis)
✗ Non-legacy/well-tested modules
✗ GUI-specific tests
✗ Performance benchmarking (only recommendations)
✗ Mutation testing details

### Modules Analyzed

**Core Legacy (GOD OBJECTS):**
1. `xpcsviewer/xpcs_file.py` - 39 public methods
2. `xpcsviewer/viewer_kernel.py` - 28 public methods

**Analysis Modules:**
3. `xpcsviewer/module/g2mod.py` - 11 functions
4. `xpcsviewer/module/twotime.py` - 4 functions
5. `xpcsviewer/module/saxs1d.py` - 10 functions
6. `xpcsviewer/module/saxs2d.py` - 1 function
7. `xpcsviewer/module/tauq.py` - 2 functions
8. `xpcsviewer/module/intt.py` - 4 functions

**FileIO (Infrastructure):**
9. `xpcsviewer/fileIO/hdf_reader.py` - 25+ methods
10. `xpcsviewer/fileIO/qmap_utils.py` - 20+ methods

---

## Q&A

### Q: Why is coverage so low despite having many tests?

**A:** The tests exist but focus on initialization and edge cases rather than the main public methods. For example:
- XpcsFile has 39 tests but 34 test initialization/memory
- Only 5 tests actually call the data access/analysis methods
- ViewerKernel has 50 tests but only test metadata management

### Q: Which tests should we prioritize?

**A:** Follow Phase 1 priority order in **PRIORITY_TEST_SPECIFICATIONS.md**:
1. G2 data access (XF-001 to XF-004)
2. Fitting methods (XF-009 to XF-012)
3. Plotting methods (VK-001 to VK-005)
4. Export methods (VK-006, VK-007)
5. ROI operations (VK-008, VK-009)

### Q: How much effort to fix this?

**A:** Approximately:
- Phase 1 (critical paths): 60 hours → 50% coverage
- Phase 2 (integration): 40 hours → 70% coverage
- Phase 3 (edge cases): 50 hours → 80% coverage
- **Total: 150 hours over 3 sprints**

### Q: Should we refactor the god objects?

**A:** Yes, but after establishing test coverage:
1. Phase 1: Establish baseline (50% coverage)
2. Phase 2: Document refactoring plan
3. Phase 3: Execute refactoring with safety net
4. Post-refactor: Achieve 80%+ coverage

### Q: How do we maintain this coverage?

**A:** See **TEST_COVERAGE_ANALYSIS.md**: Metrics & Monitoring section:
- Set minimum 70% coverage threshold in CI/CD
- Weekly coverage trending reports
- Block merge requests if coverage drops
- Mutation testing to find weak assertions

---

## Next Steps

### For Managers/Leads

1. Review **COVERAGE_SUMMARY.txt** (10 min)
2. Share with team, discuss findings (30 min meeting)
3. Decide if Phase 1 fits in next sprint (commitment: 60 hours)
4. Allocate 2-3 developers for 2 weeks

### For Developers

1. Read **PRIORITY_TEST_SPECIFICATIONS.md** for your area
2. Start with test XF-001 (basic G2 data access)
3. Follow the template: objective → test data → assertions
4. Use provided fixture patterns
5. Run tests locally before submitting PR

### For QA Engineers

1. Review **TEST_COVERAGE_ANALYSIS.md**: Risk Assessment
2. Plan manual testing for CRITICAL areas not yet automated
3. Create test matrix for edge cases (Phase 3)
4. Set up continuous coverage monitoring

---

## Support & Questions

For questions about:
- **Coverage analysis:** See TEST_COVERAGE_ANALYSIS.md
- **Test specifications:** See PRIORITY_TEST_SPECIFICATIONS.md
- **Quick overview:** See COVERAGE_SUMMARY.txt
- **Implementation approach:** Ask test automation lead

---

## Document Version & History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-06 | Test Analysis | Initial analysis - 8 modules, 64 untested methods identified |

---

**Generated:** 2026-01-06
**Analysis Scope:** Legacy component test coverage evaluation
**Repository:** xpcsviewer (branch: 001-jax-migration)
**Status:** Ready for Implementation
