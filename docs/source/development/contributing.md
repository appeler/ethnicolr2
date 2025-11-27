# Contributing to ethnicolr2

We welcome contributions to ethnicolr2! This guide will help you get started with contributing code, documentation, or reporting issues.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/appeler/ethnicolr2.git
cd ethnicolr2

# Install development dependencies with uv (recommended)
uv sync --all-groups

# Or with pip
pip install -e ".[dev,test,docs]"
```

### Development Environment

```bash
# Activate the virtual environment (if using uv)
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate     # Windows

# Verify installation
python -c "import ethnicolr2; print(ethnicolr2.__version__)"
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

### 2. Make Changes

Follow these guidelines:
- Write clear, well-documented code
- Add type hints to all functions
- Follow existing code style
- Add tests for new functionality

### 3. Code Quality

```bash
# Format code with ruff
uv run ruff format .

# Check for linting issues
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### 4. Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ethnicolr2 --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_models.py
```

### 5. Documentation

If you're changing APIs or adding features:

```bash
# Build documentation locally
cd docs && uv run make html

# View documentation
open docs/build/html/index.html  # macOS
# xdg-open docs/build/html/index.html  # Linux
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description

- Detailed explanation of changes
- Any breaking changes
- Resolves #issue-number"
```

## Contribution Types

### Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, ethnicolr2 version
2. **Minimal example**: Smallest code that reproduces the issue
3. **Expected vs actual behavior**
4. **Error messages** (full traceback if applicable)

**Template:**
```markdown
## Bug Report

**Environment:**
- OS: macOS 14.0
- Python: 3.11.5
- ethnicolr2: 0.2.0

**Code to reproduce:**
\\```python
import ethnicolr2
# Minimal example here
\\```

**Expected:** Description of expected behavior
**Actual:** Description of what actually happened
**Error:** Full error message if applicable
```

### Feature Requests

For new features, please include:

1. **Use case**: Why is this feature needed?
2. **Proposed API**: How should it work?
3. **Examples**: Show usage examples
4. **Alternatives**: Have you considered other approaches?

### Code Contributions

#### Adding New Models

If adding a new prediction model:

1. Create model class in appropriate module
2. Follow naming convention: `{Dataset}{Type}Model`
3. Inherit from `EthnicolrModelClass`
4. Add comprehensive tests
5. Update documentation

Example structure:
```python
from .ethnicolr_class import EthnicolrModelClass

class NewDatasetLstmModel(EthnicolrModelClass):
    \"\"\"LSTM model trained on new dataset.\"\"\"
    
    MAX_SEQUENCE_LENGTH = 30
    VOCAB_FN = "new_vocab.joblib"
    MODEL_FN = "new_model.pt"
    
    @classmethod
    def predict(cls, df, **kwargs):
        # Implementation
        pass
```

#### Adding New Features

For new features:
1. Add function to appropriate module
2. Include comprehensive docstring
3. Add type hints
4. Write tests covering edge cases
5. Update documentation

### Documentation Contributions

Documentation improvements are always welcome:

- **Fix typos** or unclear explanations
- **Add examples** for existing features
- **Improve API documentation**
- **Update guides** for new best practices

```bash
# Test documentation changes locally
cd docs
uv run make clean html
open build/html/index.html
```

## Code Style Guide

### Python Style

We use [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting:

- **Line length**: 88 characters (Black-compatible)
- **Import sorting**: isort-compatible
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings

### Docstring Format

```python
def example_function(name: str, threshold: float = 0.5) -> pd.DataFrame:
    \"\"\"Predict something from name.
    
    Args:
        name: The input name to process.
        threshold: Confidence threshold for predictions.
        
    Returns:
        DataFrame with predictions and confidence scores.
        
    Raises:
        ValueError: If threshold is not between 0 and 1.
        KeyError: If required columns are missing.
        
    Examples:
        >>> df = pd.DataFrame({'names': ['Smith', 'Zhang']})
        >>> result = example_function(df, threshold=0.8)
        >>> print(result.columns)
        ['names', 'race', 'confidence']
    \"\"\"
```

### Testing Guidelines

- **Test coverage**: Aim for >90% coverage
- **Test structure**: Use pytest fixtures for common setup
- **Test data**: Use small, predictable datasets
- **Edge cases**: Test empty inputs, missing columns, etc.

```python
def test_prediction_function():
    \"\"\"Test basic prediction functionality.\"\"\"
    # Arrange
    df = pd.DataFrame({'last_name': ['Smith', 'Zhang']})
    
    # Act
    result = pred_fl_last_name(df, lname_col='last_name')
    
    # Assert
    assert len(result) == 2
    assert 'race' in result.columns
    assert result['race'].isin(['asian', 'hispanic', 'nh_black', 'nh_white']).all()
```

## Release Process

### Version Bumping

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Create release commit
git commit -m "release: version 0.2.1"
git tag v0.2.1
git push origin main --tags
```

### Changelog

Update `CHANGELOG.md` with:
- **Breaking changes** (if any)
- **New features**
- **Bug fixes**
- **Technical improvements**

## Getting Help

### Development Questions

- **GitHub Discussions**: General development questions
- **GitHub Issues**: Bug reports and feature requests
- **Code Review**: Submit PRs for feedback

### Resources

- **Project Structure**: See {doc}`../getting-started/concepts`
- **API Documentation**: See {doc}`../api-reference/index`
- **Testing Guide**: See {doc}`testing`

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- GitHub contributor list
- Release notes for significant contributions

Thank you for contributing to ethnicolr2! 🎉