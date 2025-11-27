# Installation

## Requirements

- **Python**: 3.11 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM recommended

## Install from PyPI

### Using uv (Recommended)

```bash
uv add ethnicolr2
```

### Using pip

```bash
pip install ethnicolr2
```

## Install from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/appeler/ethnicolr2.git
cd ethnicolr2

# Install with uv
uv sync

# Or install with pip in development mode
pip install -e .
```

## Verify Installation

Test that the installation worked correctly:

```python
import ethnicolr2
print(ethnicolr2.__version__)

# Quick test
import pandas as pd
from ethnicolr2 import census_ln

df = pd.DataFrame({'last_name': ['Smith', 'Zhang']})
result = census_ln(df, 'last_name')
print(result)
```

## Dependencies

ethnicolr2 automatically installs these key dependencies:

- **PyTorch 2.8.0**: Neural network framework
- **pandas**: Data manipulation and analysis
- **NumPy 2.x**: Numerical computing
- **scikit-learn 1.5.1**: Machine learning utilities
- **joblib**: Model serialization

## Development Dependencies

For contributors and developers:

```bash
# Install all development dependencies
uv sync --all-groups

# Install specific groups
uv sync --group test     # Testing tools
uv sync --group dev      # Development tools  
uv sync --group docs     # Documentation tools
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'numpy.exceptions'**
```bash
# Update to NumPy 2.x
pip install --upgrade "numpy>=2.3.5"
```

**CUDA/GPU Issues**
```python
# Check if CUDA is available
import torch
print(torch.cuda.is_available())
```

**Memory Issues**
- Ensure you have at least 4GB RAM
- Models are loaded on-demand to minimize memory usage

### Getting Help

- [GitHub Issues](https://github.com/appeler/ethnicolr2/issues)
- [GitHub Discussions](https://github.com/appeler/ethnicolr2/discussions)

## What's Next?

- {doc}`quickstart`: Learn the basics in 5 minutes
- {doc}`concepts`: Understand the key concepts
- {doc}`../user-guide/examples`: See practical examples