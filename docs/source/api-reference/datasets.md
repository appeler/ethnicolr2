# Datasets API Reference

This page documents dataset-related classes and utilities for handling name data in ethnicolr2.

## Core Dataset Classes

### EthniDataset

```{eval-rst}
.. autoclass:: ethnicolr2.dataset.EthniDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
```

## Usage Examples

### Basic Dataset Usage

```python
from ethnicolr2.dataset import EthniDataset
import torch

# Create dataset from list of names
names = ['Smith', 'Zhang', 'Rodriguez']
dataset = EthniDataset(names, max_length=30)

print(f"Dataset size: {len(dataset)}")
print(f"First item: {dataset[0]}")  # Tensor of character indices
```

### DataLoader Integration

```python
from ethnicolr2.dataset import EthniDataset
import torch

# Create dataset
names = ['Smith', 'Zhang', 'Rodriguez']
dataset = EthniDataset(names, max_length=30)

# Create DataLoader for batch processing
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0  # Set to 0 for simple character data
)

# Iterate through batches
for batch in dataloader:
    print(batch.shape)  # [batch_size, max_length]
    print(batch.dtype)  # torch.int64
```

### Custom Character Mapping

```python
from ethnicolr2.dataset import EthniDataset

# Create dataset with custom parameters
dataset = EthniDataset(
    names=['José', 'O\'Connor', 'van der Berg'],
    max_length=20
)

# Access the character vocabulary
for i, name in enumerate(['José', 'O\'Connor', 'van der Berg']):
    tensor = dataset[i]
    print(f"'{name}' -> {tensor[:10]}...")  # First 10 character indices
```

## Data Processing Pipeline

The `EthniDataset` class handles the conversion from raw names to tensors suitable for neural network training:

1. **Character Mapping**: Each character is mapped to a unique integer ID
2. **Sequence Padding**: Names are padded or truncated to a fixed length
3. **Tensor Conversion**: Strings become integer tensors for PyTorch

### Character Set

The dataset uses a standard ASCII character set with special handling for:
- Uppercase and lowercase letters (A-Z, a-z)
- Numbers (0-9)
- Common punctuation and symbols
- Spaces and special characters in names

## Performance Considerations

### Memory Usage

```python
# For large datasets, consider batch loading
def create_batched_dataset(names, batch_size=1000):
    for i in range(0, len(names), batch_size):
        batch_names = names[i:i+batch_size]
        yield EthniDataset(batch_names, max_length=30)

# Usage
large_name_list = ['Smith'] * 10000  # Large list
for batch_dataset in create_batched_dataset(large_name_list):
    # Process each batch
    loader = torch.utils.data.DataLoader(batch_dataset, batch_size=32)
    for batch in loader:
        # Your processing here
        pass
```
