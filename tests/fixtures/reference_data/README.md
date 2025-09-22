# Reference Data for Scientific Validation

This directory contains validated reference datasets used for scientific algorithm validation in the XPCS Toolkit. All data has been carefully validated against theoretical models, analytical solutions, or established reference implementations.

## Directory Structure

### `/g2_analysis/`
Contains reference G2 correlation function data with known analytical properties:
- `single_exponential_*.npz`: Single exponential decay data with varying parameters
- `double_exponential_*.npz`: Double exponential decay systems
- `stretched_exponential_*.npz`: Stretched exponential (KWW) decay data
- `analytical_solutions.py`: Exact analytical solutions for validation

### `/saxs_analysis/`
Contains reference SAXS scattering data:
- `sphere_form_factors_*.npz`: Sphere form factor data for various radii
- `power_law_scattering_*.npz`: Power law scattering data
- `guinier_approximation_*.npz`: Data for Guinier analysis validation
- `polydisperse_systems_*.npz`: Polydisperse particle systems
- `experimental_benchmarks/`: Validated experimental data

### `/twotime_analysis/`
Contains reference two-time correlation data:
- `brownian_motion_*.npz`: Two-time correlations for Brownian motion
- `subdiffusion_*.npz`: Subdiffusive two-time correlations  
- `correlation_matrices_*.npz`: Validated correlation matrices
- `stationary_processes_*.npz`: Stationary process examples

### `/diffusion_analysis/`
Contains reference tau-Q diffusion data:
- `brownian_diffusion_*.npz`: Simple Brownian diffusion data
- `subdiffusion_*.npz`: Subdiffusive motion data
- `cooperative_diffusion_*.npz`: Systems with hydrodynamic interactions
- `polydisperse_diffusion_*.npz`: Polydisperse particle diffusion

## Data Format

All reference data files are stored in NumPy `.npz` format and contain:

### Required Fields
- `x_values`: Independent variable values (time, q-values, etc.)
- `y_values`: Dependent variable values (G2, intensity, etc.)
- `y_errors`: Estimated uncertainties (if available)
- `metadata`: Dictionary with creation parameters and validation info

### Metadata Fields
- `creation_date`: ISO format date of data creation
- `validation_method`: Method used to validate the data
- `theoretical_params`: Known theoretical parameters
- `numerical_precision`: Estimated numerical precision
- `reference_source`: Literature reference or analytical formula
- `notes`: Additional validation notes

## Validation Criteria

All reference data must meet these criteria:

1. **Theoretical Accuracy**: Data must match analytical solutions within numerical precision
2. **Physical Validity**: All physical constraints must be satisfied
3. **Statistical Properties**: Statistical properties must be correct
4. **Documentation**: Complete metadata and validation documentation
5. **Reproducibility**: Generation process must be documented and reproducible

## Usage in Tests

Reference data is loaded using the `load_reference_data()` function:

```python
from tests.fixtures.reference_data import load_reference_data

# Load G2 single exponential data
data = load_reference_data('g2_analysis', 'single_exponential_standard')
x, y, y_err, metadata = data['x_values'], data['y_values'], data['y_errors'], data['metadata']

# Validate against expected parameters
expected_params = metadata['theoretical_params']
```

## Adding New Reference Data

To add new reference data:

1. Generate data using validated theoretical models
2. Verify against analytical solutions where possible
3. Include complete metadata
4. Add validation tests in corresponding test modules
5. Document the validation process

## Maintenance

Reference data should be periodically reviewed and updated:
- Verify continued accuracy with improved algorithms
- Add new reference cases as needed
- Update documentation and metadata
- Ensure compatibility with current data formats

## External References

Some reference data is derived from:
- NIST Standard Reference Materials
- Published XPCS datasets with known properties
- Theoretical calculations from literature
- Cross-validation with other software packages

All external sources are properly cited in the metadata.
