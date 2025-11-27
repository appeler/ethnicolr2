# Testing Guide

This guide covers running and writing tests for ethnicolr2.

## Running Tests

### All Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest --verbose

# Run with coverage
uv run pytest --cov=ethnicolr2 --cov-report=term-missing
```

### Specific Tests

```bash
# Run specific test file
uv run pytest tests/test_models.py

# Run specific test class
uv run pytest tests/test_models.py::TestFloridaModels

# Run specific test method
uv run pytest tests/test_models.py::TestFloridaModels::test_florida_last_name_model
```

## Test Organization

Tests are organized in the `tests/` directory:

- `test_census_lookup.py`: Census data lookup tests
- `test_models.py`: Model loading and prediction tests  
- `test_cli.py`: Command-line interface tests
- `test_validation.py`: Input validation tests
- `test_regression.py`: Regression tests for bug fixes

## Writing Tests

### Test Structure

```python
import unittest
import pandas as pd
from ethnicolr2 import pred_fl_last_name

class TestPredictions(unittest.TestCase):
    def setUp(self):
        \"\"\"Set up test fixtures.\"\"\"
        self.test_df = pd.DataFrame({
            'last_name': ['Smith', 'Zhang', 'Rodriguez']
        })
    
    def test_basic_prediction(self):
        \"\"\"Test basic prediction functionality.\"\"\"
        result = pred_fl_last_name(self.test_df, lname_col='last_name')
        
        # Check output structure
        self.assertEqual(len(result), 3)
        self.assertIn('race', result.columns)
        self.assertIn('asian', result.columns)
        
        # Check data types
        self.assertTrue(result['asian'].dtype == 'float64')
        
        # Check value ranges
        self.assertTrue((result['asian'] >= 0).all())
        self.assertTrue((result['asian'] <= 1).all())
```

### Coverage Requirements

Aim for >90% test coverage:

```bash
# Generate detailed coverage report
uv run pytest --cov=ethnicolr2 --cov-report=html
open htmlcov/index.html
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled daily runs

See `.github/workflows/ci.yml` for configuration.

## Test Data

Tests use small, predictable datasets to ensure:
- Fast execution
- Deterministic results  
- Easy debugging

For more details, see {doc}`contributing`.