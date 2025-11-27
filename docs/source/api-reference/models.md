# Models API Reference

This page documents all model-related functions and classes in ethnicolr2.

## Prediction Functions

### Census Models

```{eval-rst}
.. autofunction:: ethnicolr2.census_ln
   :noindex:
```

```{eval-rst}
.. autofunction:: ethnicolr2.pred_census_last_name
   :noindex:
```

### Florida Models

```{eval-rst}
.. autofunction:: ethnicolr2.pred_fl_last_name
   :noindex:
```

```{eval-rst}
.. autofunction:: ethnicolr2.pred_fl_full_name
   :noindex:
```

## Neural Network Classes

### LSTM Model

```{eval-rst}
.. autoclass:: ethnicolr2.models.LSTM
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
```

### Base Model Class

```{eval-rst}
.. autoclass:: ethnicolr2.ethnicolr_class.EthnicolrModelClass
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
```

### Model Implementation Notes

The prediction functions above are implemented using internal model classes that handle:

- **Model Loading**: Automatic loading of pre-trained PyTorch models
- **Text Processing**: Character-level encoding and sequence preparation  
- **Batch Inference**: Efficient processing of multiple names
- **Result Formatting**: Converting raw model outputs to readable predictions

For most use cases, use the high-level prediction functions rather than the internal model classes directly.

## Examples

### Loading Models Manually

```python
from ethnicolr2.pred_fl_ln_lstm import LastNameLstmModel

# Load Florida last name model
model = LastNameLstmModel()

# Make predictions
import pandas as pd
df = pd.DataFrame({'last_name': ['Smith', 'Zhang']})
result = model.predict(df, vocab_fn=model.VOCAB_FN, model_fn=model.MODEL_FN)
print(result)
```

### Custom Model Parameters

```python
from ethnicolr2.models import LSTM
import torch

# Create custom LSTM
model = LSTM(
    vocab_size=128,      # Character vocabulary size
    hidden_size=256,     # LSTM hidden units
    num_layers=2,        # Number of LSTM layers
    num_classes=5,       # Number of output classes
    dropout=0.2          # Dropout rate
)

# Example forward pass
batch_size, sequence_length = 32, 20
input_tensor = torch.randint(0, 128, (batch_size, sequence_length))
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # [32, 5]
```

### Model Configuration

The prediction models use different configuration parameters:

| Model | Max Length | Character Set | Training Data |
|-------|------------|---------------|---------------|
| Census Last Name | 15 chars | ASCII + common punctuation | US Census 2000/2010 |
| Florida Last Name | 30 chars | Extended character set | FL voter registration |
| Florida Full Name | 47 chars | Extended character set | FL voter registration |

All models use 2-layer LSTM networks with 256 hidden units and batch processing for efficiency.