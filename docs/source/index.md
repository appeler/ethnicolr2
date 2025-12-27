# ethnicolr2: Predict Race and Ethnicity From Names

![CI](https://github.com/appeler/ethnicolr2/actions/workflows/ci.yml/badge.svg)
![PyPI Version](https://img.shields.io/pypi/v/ethnicolr2.svg)
![Python Version](https://img.shields.io/pypi/pyversions/ethnicolr2.svg)
![Downloads](https://pepy.tech/badge/ethnicolr2)

**ethnicolr2** is a modern PyTorch-based machine learning package that predicts race and ethnicity from names using LSTM neural networks. It's trained on US Census data and Florida voter registration data to provide accurate predictions based on:

- **Last name only** (census model or Florida model)
- **First and last name combined** (Florida full name model)

## Quick Start

```bash
# Install ethnicolr2
uv add ethnicolr2
# or
pip install ethnicolr2
```

```python
import pandas as pd
from ethnicolr2 import pred_fl_last_name

# Predict from last names
df = pd.DataFrame({'last': ['Smith', 'Zhang', 'Rodriguez']})
result = pred_fl_last_name(df, lname_col='last')
print(result)
```

## Key Features

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🎯 High Accuracy
:link: getting-started/concepts
:link-type: doc

Trained on US Census data and Florida voter registration with proven accuracy for demographic prediction.
:::

:::{grid-item-card} ⚡ Modern PyTorch
:link: api-reference/index
:link-type: doc

Built with PyTorch 2.x for efficient neural network inference with LSTM models.
:::

:::{grid-item-card} 🔧 Easy Integration
:link: user-guide/cli-usage
:link-type: doc

Both Python API and command-line interface for seamless integration into your workflow.
:::

::::

## Documentation Sections

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 Getting Started
:link: getting-started/installation
:link-type: doc

Installation, quickstart guide, and core concepts to get you up and running quickly.
:::

:::{grid-item-card} 📓 Examples
:link: examples/index
:link-type: doc

Interactive Jupyter notebooks demonstrating real-world applications and best practices.
:::

:::{grid-item-card} 📚 User Guide
:link: user-guide/census-data
:link-type: doc

Detailed tutorials and best practices for different use cases.
:::

:::{grid-item-card} 📖 API Reference
:link: api-reference/index
:link-type: doc

Complete API documentation with all classes, functions, and parameters.
:::

:::{grid-item-card} 🛠️ Development
:link: development/contributing
:link-type: doc

Contributing guidelines, testing, and development setup information.
:::

::::

## Supported Prediction Categories

The models predict one of five race/ethnicity categories:
- `nh_white`: Non-Hispanic White
- `nh_black`: Non-Hispanic Black
- `hispanic`: Hispanic
- `asian`: Asian
- `other`: Other

## Available Models

| Model | Input | Training Data | Use Case |
|-------|--------|---------------|----------|
| Census Last Name | Last name only | US Census 2000/2010 | General population predictions |
| Florida Last Name | Last name only | FL voter registration | State-specific predictions |
| Florida Full Name | First + Last name | FL voter registration | Highest accuracy predictions |

```{toctree}
:maxdepth: 2
:hidden:

getting-started/installation
getting-started/quickstart
getting-started/concepts
```

```{toctree}
:caption: Examples
:maxdepth: 2
:hidden:

examples/index
```

```{toctree}
:caption: User Guide
:maxdepth: 2
:hidden:

user-guide/census-data
user-guide/florida-models
user-guide/cli-usage
```

```{toctree}
:caption: API Reference
:maxdepth: 2
:hidden:

api-reference/index
```

```{toctree}
:caption: Development
:maxdepth: 2
:hidden:

development/contributing
development/testing
development/changelog
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
