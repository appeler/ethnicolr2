# Datasets API Reference

This page documents dataset-related classes and utilities.

## Dataset Classes

```{eval-rst}
.. autoclass:: ethnicolr2.dataset.EthniDataset
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

```python
from ethnicolr2.dataset import EthniDataset
import torch

# Create dataset
names = ['Smith', 'Zhang', 'Rodriguez'] 
dataset = EthniDataset(names, max_length=30)

# Create DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Iterate through batches
for batch in dataloader:
    print(batch.shape)  # [batch_size, max_length]
```