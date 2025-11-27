# API Reference

Complete reference for all ethnicolr2 classes, functions, and modules.

## Quick Reference

### Prediction Functions

```python
from ethnicolr2 import (
    # Census models
    census_ln,
    pred_census_last_name,
    
    # Florida models  
    pred_fl_last_name,
    pred_fl_full_name
)
```

### Core Classes

```python
from ethnicolr2.models import LSTM
from ethnicolr2.dataset import EthniDataset
from ethnicolr2.ethnicolr_class import EthnicolrModelClass
```

## Module Overview

### Prediction Models

For detailed documentation of prediction functions:

- **{doc}`models`**: Complete model API including LSTM implementations and prediction functions
- **{doc}`datasets`**: Data handling classes and utilities  
- **{doc}`utilities`**: Command-line tools and helper functions

### Quick Function Reference

| Function | Purpose | Input |
|----------|---------|-------|
| `pred_fl_last_name` | Florida last name model | Last name column |
| `pred_fl_full_name` | Florida full name model | First + last name columns |
| `pred_census_last_name` | Census last name model | Last name column |
| `census_ln` | Census data lookup | Last name column |

See {doc}`models` for complete function signatures and examples.

## Usage Patterns

### Basic Prediction

```python
import pandas as pd
from ethnicolr2 import pred_fl_last_name

# Create DataFrame
df = pd.DataFrame({'names': ['Smith', 'Zhang', 'Rodriguez']})

# Get predictions
result = pred_fl_last_name(df, lname_col='names')

# Access predictions
print(result['preds'])       # Predicted categories
print(result['probs'])       # Probability distributions
```

### Advanced Usage

```python
from ethnicolr2.models import LSTM
from ethnicolr2.dataset import EthniDataset
import torch

# Load custom model
model = LSTM(vocab_size=100, hidden_size=256, num_classes=5)
model.load_state_dict(torch.load('custom_model.pt'))

# Create dataset
dataset = EthniDataset(names=['Smith', 'Zhang'], max_length=30)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Custom inference
model.eval()
with torch.no_grad():
    for batch in dataloader:
        outputs = model(batch)
        predictions = torch.softmax(outputs, dim=1)
```

## Error Handling

All functions provide clear error messages for common issues:

```python
try:
    result = pred_fl_last_name(df, lname_col='nonexistent_column')
except KeyError as e:
    print(f"Column error: {e}")

try:
    result = pred_fl_last_name("not_a_dataframe", lname_col='names')
except TypeError as e:
    print(f"Type error: {e}")
```

## Performance Considerations

### Memory Usage

- Models are loaded on-demand to minimize memory usage
- Large datasets are processed in batches automatically
- GPU acceleration is used when available

### Batch Processing

```python
# Efficient processing of large datasets
def process_large_dataset(df: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df[i:i+chunk_size]
        chunk_result = pred_fl_last_name(chunk, lname_col='names')
        results.append(chunk_result)
    return pd.concat(results, ignore_index=True)
```

## Next Sections

```{toctree}
:maxdepth: 2

models
datasets
utilities
```