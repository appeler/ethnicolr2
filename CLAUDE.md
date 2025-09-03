# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ethnicolr2 is a PyTorch-based machine learning package that predicts race and ethnicity from names using LSTM neural networks. It's a modernized version of the original ethnicolr package, trained on US Census data and Florida voter registration data. The package supports prediction based on:

- Last name only (census model or Florida model)
- First and last name combined (Florida full name model)

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with testing dependencies
pip install -e ".[test]"

# Install with documentation dependencies
pip install -e ".[docs]"

# Install all optional dependencies
pip install -e ".[all]"
```

### Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest --verbose

# Run specific test file
pytest ethnicolr2/tests/test_010_census_ln.py

# Run tests with coverage
coverage run -m pytest && coverage report
```

### Code Quality
```bash
# Format code with black
black ethnicolr2/

# Sort imports with isort
isort ethnicolr2/

# Check code style with flake8
flake8 ethnicolr2/
```

### Documentation
```bash
# Build documentation (from docs/ directory)
cd docs && make html

# View documentation
open docs/_build/html/index.html
```

## Code Architecture

### Core Components

1. **Model Classes** (`ethnicolr_class.py`):
   - `EthnicolrModelClass`: Base class providing common prediction functionality
   - Contains shared methods for data preprocessing, tensor conversion, and model inference
   - Handles batch processing with PyTorch DataLoader

2. **Neural Network Models** (`models.py`):
   - `LSTM`: Custom LSTM implementation with embedding layer
   - Configured for character-level sequence processing
   - Outputs log-softmax probabilities for race/ethnicity categories

3. **Prediction Modules**:
   - `pred_fl_ln_lstm.py`: Florida last name model (`LastNameLstmModel`)
   - `pred_fl_fn_lstm.py`: Florida full name model (`FullNameLstmModel`)
   - `pred_cen_ln_lstm.py`: Census last name model (`CensusLstmModel`)

4. **Data Handling**:
   - `dataset.py`: `EthniDataset` class for PyTorch data loading
   - `census_ln.py`: Census data lookup functionality
   - `utils.py`: Command-line argument parsing utilities

### Model Files and Data

- **Pre-trained Models** (in `models/`):
  - `lstm_lastname_gen.pt`: Florida last name LSTM model
  - `lstm_fullname.pt`: Florida full name LSTM model
  - `census_lstm_lastname.pt`: Census last name LSTM model
  - Corresponding `.joblib` vectorizer files for each model

- **Training Data** (in `data/`):
  - Census surname data from 2000 and 2010
  - Florida voter registration data samples
  - Example input files for testing

### Prediction Categories

The models predict one of five race/ethnicity categories:
- `nh_white`: Non-Hispanic White
- `nh_black`: Non-Hispanic Black
- `hispanic`: Hispanic
- `asian`: Asian
- `other`: Other

### Model-Specific Parameters

- **Florida models**: 30-character max name length, 256 hidden units
- **Census models**: 15-character max name length, different category ordering
- **Full name models**: 47-character max name length
- All models use 2-layer LSTM with batch size 64

### Key Dependencies

- **PyTorch 2.7.0**: Neural network framework (exact version required)
- **scikit-learn 1.5.1**: For vectorizers (exact version required for model compatibility)
- **pandas>=1.3.0**: Data manipulation
- **joblib==1.3.1**: Model serialization (exact version required)
- **tqdm==4.66.3**: Progress bars (exact version required)
- **numpy>=1.20.0,<2.0.0**: Numerical computing

## Testing Strategy

Tests are organized by functionality:
- `test_010_census_ln.py`: Census data lookup tests
- `test_020_pred_census_ln.py`: Census prediction model tests
- `test_040_pred_fl.py`: Florida model tests
- `test_060_pred.py`: General prediction tests
- `test_cli.py`: Command-line interface tests
- `test_validation.py`: Input validation tests
- `test_bug_fixes.py`: Regression tests for bug fixes

All tests use unittest framework and include data validation, prediction accuracy checks, and error handling verification.

## Command Line Interface

The package provides CLI scripts for each prediction model:

```bash
# Census last name prediction
census_ln input.csv -l last_name -o output.csv

# Florida last name prediction
pred_fl_last_name input.csv -l last_name -o output.csv

# Florida full name prediction (requires both first and last name)
pred_fl_full_name input.csv -l last_name -f first_name -o output.csv

# Census last name prediction with LSTM
pred_census_last_name input.csv -l last_name -o output.csv

# Download pre-trained models
ethnicolr2_download_models
```