#!/usr/bin/env bash
# Re-run all baseline benchmarks and collect results for before/after comparison.
# Usage: bash tests/benchmarks/performance/rerun_baselines.sh [output_dir]
#
# Outputs:
#   <output_dir>/benchmark_timing.txt     — pytest-benchmark table
#   <output_dir>/cold_warm_memory.txt     — cold/warm/memory measurements
#   <output_dir>/cprofile_hotpaths.txt    — cProfile call graph summaries

set -euo pipefail

OUTPUT_DIR="${1:-tests/reports/post_optimization}"
mkdir -p "$OUTPUT_DIR"

export JAX_PLATFORMS=cpu

echo "=== Running pytest-benchmark timing ==="
uv run pytest tests/benchmarks/performance/test_hotpath_baseline.py \
  --benchmark-only \
  --benchmark-columns=mean,stddev,min,max,rounds \
  --benchmark-sort=mean \
  -m "not slow" \
  --timeout=300 \
  --no-cov \
  2>&1 | tee "$OUTPUT_DIR/benchmark_timing.txt"

echo ""
echo "=== Running cold/warm and memory tests ==="
uv run pytest tests/benchmarks/performance/test_hotpath_baseline.py \
  -v -s \
  --benchmark-disable \
  -m "not slow" \
  --timeout=300 \
  --no-cov \
  -k "cold_warm or memory" \
  2>&1 | tee "$OUTPUT_DIR/cold_warm_memory.txt"

echo ""
echo "=== Running cProfile analysis ==="
uv run python tests/benchmarks/performance/_cprofile_hotpaths.py \
  2>&1 | tee "$OUTPUT_DIR/cprofile_hotpaths.txt"

echo ""
echo "=== Running full test suite (regression check) ==="
uv run pytest tests/ --timeout=120 -q --no-cov \
  2>&1 | tee "$OUTPUT_DIR/test_suite_results.txt"

echo ""
echo "=== Running numerical accuracy tests ==="
uv run pytest tests/jax_migration/numerical/ tests/jax_migration/precision/ -v --no-cov \
  2>&1 | tee "$OUTPUT_DIR/numerical_accuracy.txt"

echo ""
echo "All results saved to $OUTPUT_DIR/"
echo "Compare with tests/reports/baseline_profile.md for before/after analysis."
