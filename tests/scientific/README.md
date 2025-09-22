# Scientific Algorithm Validation Framework

This directory contains a comprehensive scientific validation framework for the XPCS Toolkit. The framework ensures that all scientific algorithms meet rigorous mathematical, physical, and statistical standards.

## Overview

The scientific validation framework consists of three main components:

1. **Algorithm Validation** (`algorithms/`) - Tests specific XPCS algorithms for correctness
2. **Property Validation** (`properties/`) - Verifies mathematical and physical properties  
3. **Reference Validation** (`reference_validation/`) - Cross-validates against established references

## Quick Start

Run the complete validation suite:

```bash
# Run all validation tests
python -m tests.framework.runners.run_scientific_validation --generate-report

# Run quick validation (subset of tests)
python -m tests.framework.runners.run_scientific_validation --quick

# Run specific algorithm tests
python -m tests.framework.runners.run_scientific_validation --suite algorithms --algorithm g2
```

## Framework Components

### Algorithm Validation (`algorithms/`)

Tests individual algorithms for mathematical correctness and physical validity:

- **G2 Analysis** (`test_g2_analysis.py`): Correlation function properties, fitting algorithms, statistical accuracy
- **SAXS Analysis** (`test_saxs_analysis.py`): Scattering calculations, form factors, intensity normalization
- **Two-Time Analysis** (`test_twotime_analysis.py`): Correlation matrix properties, causality constraints
- **Diffusion Analysis** (`test_diffusion_analysis.py`): Tau-Q relationships, physical constraints, power-law fitting
- **Fitting Algorithms** (`test_fitting_algorithms.py`): Statistical validation, convergence analysis, error estimation

### Property Validation (`properties/`)

Validates fundamental mathematical and physical properties:

- **Mathematical Invariants** (`mathematical_invariants.py`): Core mathematical properties (normalization, monotonicity, etc.)
- **Statistical Properties** (`statistical_properties.py`): Parameter estimation, uncertainty quantification, statistical tests
- **Physical Constraints** (`physical_constraints.py`): Physical laws (Stokes-Einstein, thermodynamics, conservation laws)

### Reference Validation (`reference_validation/`)

Cross-validates against established references and benchmarks:

- **Cross-Validation Framework** (`cross_validation_framework.py`): K-fold validation, bootstrap methods, reference comparison
- **Analytical Benchmarks** (`analytical_benchmarks.py`): Exact analytical solutions for validation
- **Reference Implementations** (`reference_implementations.py`): Comparison with literature and software packages

## Test Categories

### Property-Based Testing

Uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing:

```python
@given(
    beta=st.floats(min_value=0.1, max_value=2.0),
    gamma=st.floats(min_value=1.0, max_value=10000.0)
)
def test_g2_properties(self, beta, gamma):
    """Test G2 properties across parameter ranges"""
    # Generate G2 data with given parameters
    # Verify mathematical properties hold
```

### Cross-Validation

Validates against multiple reference sources:

```python
# Analytical validation
analytical_validator = AnalyticalValidator(tolerance=1e-6)

# Literature reference validation  
literature_validator = LiteratureReferenceValidator()

# Software package validation
matlab_validator = SoftwarePackageValidator('matlab_dls')
```

### Statistical Validation

Ensures statistical correctness of fitting algorithms:

- Parameter uncertainty scaling (σ ∝ 1/√N)
- Chi-squared distribution of residuals
- Bootstrap confidence intervals
- Monte Carlo convergence analysis

## Reference Data

The framework includes validated reference datasets (`../fixtures/reference_data/`):

- **G2 Analysis**: Single/double/stretched exponential decays
- **SAXS Analysis**: Sphere/cylinder form factors, power law scattering
- **Diffusion Analysis**: Brownian and subdiffusive motion
- **Two-Time Analysis**: Correlation matrices for various processes

All reference data includes:
- Theoretical parameters
- Validation methodology  
- Numerical precision estimates
- Literature citations

## Mathematical Properties Validated

### G2 Correlation Functions

1. **Normalization**: G2(τ=0) ≥ 1
2. **Asymptotic Behavior**: G2(τ→∞) → 1  
3. **Monotonicity**: Single exponential decay
4. **Causality**: Time-reversal symmetry for stationary processes
5. **Statistical Properties**: Proper error propagation

### SAXS Scattering

1. **Positivity**: I(q) ≥ 0 everywhere
2. **Form Factor Properties**: Oscillations, forward scattering maximum
3. **Guinier Approximation**: I(q) ∝ exp(-q²Rg²/3) for small q
4. **Porod Law**: I(q) ∝ q⁻⁴ for large q
5. **Conservation Laws**: Photon number conservation

### Diffusion Analysis

1. **Positivity**: τ(q) > 0 for all q
2. **Brownian Scaling**: τ(q) = 1/(Dq²)
3. **Subdiffusion**: τ(q) ∝ q⁻α with 0 < α < 2
4. **Stokes-Einstein Relation**: D = kT/(6πηr)
5. **Physical Bounds**: Reasonable D values for given particle sizes

### Two-Time Correlations

1. **Symmetry**: C(t₁,t₂) = C(t₂,t₁) for stationary processes
2. **Positive Definiteness**: All eigenvalues ≥ 0
3. **Diagonal Properties**: C(t,t) = intensity²(t)
4. **Causality**: Future doesn't affect past
5. **Ergodicity**: Time averages equal ensemble averages

## Physical Constraints Validated

### Thermodynamic Constraints

- Temperature ranges (1 K < T < 1000 K for typical XPCS)
- Energy scales comparable to or larger than kT
- Time scales appropriate for XPCS (µs to s)

### Scattering Physics

- Q-value ranges consistent with X-ray wavelength
- Intensity ranges realistic for photon counting
- Detector geometry consistency
- Elastic scattering constraints

### Diffusion Physics

- Stokes-Einstein relation validation
- Reasonable diffusion coefficients for particle sizes
- Temperature and viscosity dependencies
- Correlation time scales

## Error Handling and Edge Cases

The framework tests robust handling of:

- **Numerical Edge Cases**: Very small/large values, near-zero denominators
- **Statistical Edge Cases**: Limited data, high noise levels, fitting failures
- **Physical Edge Cases**: Extreme parameter values, boundary conditions
- **Data Quality Issues**: Missing data, NaN values, inconsistent arrays

## Performance Validation

Validates that optimizations preserve accuracy:

- **Vectorized Operations**: Results identical to non-vectorized versions
- **Numerical Stability**: Consistent results across parameter ranges  
- **Memory Management**: No degradation with large datasets
- **Computational Complexity**: Scaling behavior as expected

## Usage Examples

### Validate Specific Algorithm

```python
from tests.scientific.algorithms.test_g2_analysis import TestG2MathematicalProperties

# Run G2 validation tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestG2MathematicalProperties)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
```

### Property-Based Testing

```python
from tests.scientific.properties.mathematical_invariants import verify_g2_normalization

# Test G2 normalization property
tau = np.logspace(-6, 1, 100)
g2_data = 1.0 + 0.8 * np.exp(-1000 * tau)

is_valid, message = verify_g2_normalization(g2_data, tau)
assert is_valid, f"G2 normalization failed: {message}"
```

### Cross-Validation

```python
from tests.scientific.reference_validation.cross_validation_framework import ComprehensiveCrossValidationFramework

framework = ComprehensiveCrossValidationFramework()
framework.add_validator('analytical', AnalyticalValidator())

# Define test case and run validation
test_cases = [...]  # Your test cases
report = framework.run_comprehensive_validation('MyAlgorithm', test_cases)
```

### Analytical Benchmarks

```python
from tests.scientific.reference_validation.analytical_benchmarks import AnalyticalBenchmarkSuite

benchmark_suite = AnalyticalBenchmarkSuite()

# Evaluate specific benchmark
domain, values = benchmark_suite.evaluate_benchmark('sphere_form_factor')

# Validate properties
properties = benchmark_suite.validate_benchmark_properties('sphere_form_factor', domain, values)
```

## Continuous Integration

The framework integrates with CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run Scientific Validation
  run: |
    python -m tests.framework.runners.run_scientific_validation --generate-report
    # Upload validation report as artifact
```

## Extending the Framework

### Adding New Algorithm Tests

1. Create test class inheriting from `unittest.TestCase`
2. Implement property-based tests using Hypothesis
3. Add mathematical property validations
4. Include edge case and error handling tests

### Adding New Properties

1. Implement verification function in appropriate properties module
2. Return `(is_valid, error_message)` tuple
3. Include comprehensive documentation
4. Add property to relevant algorithm tests

### Adding New Benchmarks

1. Implement analytical function
2. Define default parameters and domain
3. Specify properties to validate
4. Add to benchmark suite

## Literature References

The validation framework is based on established scientific principles from:

- Lumma et al., "Dynamic light scattering in confined fluids" (2000)
- Fluerasu et al., "X-ray intensity fluctuation spectroscopy" (2007)
- Ponmurugan et al., "Analytical form factors for crystalline spheres" (2009)
- Brown, "Dynamic Light Scattering: The Method and Some Applications" (1993)

## Validation Certificate

Upon successful completion, the framework generates a validation certificate documenting:

- All tests passed
- Reference validations confirmed  
- Mathematical properties verified
- Physical constraints satisfied
- Statistical requirements met

This certificate provides confidence that the XPCS Toolkit meets rigorous scientific standards for publication-quality research.
