# Models API Reference

This page documents all model-related functions and classes in ethnicolr2.

## Prediction Functions

### Census Models

```{eval-rst}
.. autofunction:: ethnicolr2.census_ln
```

```{eval-rst}
.. autofunction:: ethnicolr2.pred_census_last_name
```

### Florida Models

```{eval-rst}
.. autofunction:: ethnicolr2.pred_fl_last_name
```

```{eval-rst}
.. autofunction:: ethnicolr2.pred_fl_full_name
```

## Neural Network Classes

### LSTM Model

```{eval-rst}
.. autoclass:: ethnicolr2.models.LSTM
   :members:
   :undoc-members:
   :show-inheritance:
```

### Base Model Class

```{eval-rst}
.. autoclass:: ethnicolr2.ethnicolr_class.EthnicolrModelClass
   :members:
   :undoc-members:
   :show-inheritance:
```

### Model Implementations

#### Census Last Name LSTM

```{eval-rst}
.. autoclass:: ethnicolr2.pred_cen_ln_lstm.CensusLstmModel
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Florida Last Name LSTM

```{eval-rst}
.. autoclass:: ethnicolr2.pred_fl_ln_lstm.LastNameLstmModel
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Florida Full Name LSTM  

```{eval-rst}
.. autoclass:: ethnicolr2.pred_fl_fn_lstm.FullNameLstmModel
   :members:
   :undoc-members:
   :show-inheritance:
```

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

### Model Constants

```{eval-rst}
.. autodata:: ethnicolr2.pred_fl_ln_lstm.LastNameLstmModel.MAX_SEQUENCE_LENGTH
.. autodata:: ethnicolr2.pred_fl_ln_lstm.LastNameLstmModel.VOCAB_FN  
.. autodata:: ethnicolr2.pred_fl_ln_lstm.LastNameLstmModel.MODEL_FN
```

```{eval-rst}
.. autodata:: ethnicolr2.pred_fl_fn_lstm.FullNameLstmModel.MAX_SEQUENCE_LENGTH
.. autodata:: ethnicolr2.pred_fl_fn_lstm.FullNameLstmModel.VOCAB_FN
.. autodata:: ethnicolr2.pred_fl_fn_lstm.FullNameLstmModel.MODEL_FN
```

```{eval-rst}
.. autodata:: ethnicolr2.pred_cen_ln_lstm.CensusLstmModel.MAX_SEQUENCE_LENGTH
.. autodata:: ethnicolr2.pred_cen_ln_lstm.CensusLstmModel.VOCAB_FN
.. autodata:: ethnicolr2.pred_cen_ln_lstm.CensusLstmModel.MODEL_FN
```