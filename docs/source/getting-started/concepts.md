# Key Concepts

Understanding the core concepts behind ethnicolr2 will help you choose the right model and interpret results effectively.

## Models and Datasets

ethnicolr2 provides three main prediction models, each trained on different datasets:

### Census Models

**Training Data**: US Census 2000 and 2010 surname statistics
**Input**: Last name only
**Categories**: 5 standard racial/ethnic categories

```python
from ethnicolr2 import census_ln, pred_census_last_name

# Census statistics (no ML prediction)
stats = census_ln(df, 'last_name', year=2010)

# Census-trained LSTM predictions  
predictions = pred_census_last_name(df, 'last_name', year=2010)
```

**Use Cases**:
- General population analysis
- Historical comparisons (2000 vs 2010)
- When you only have last names
- Academic research requiring census-based validation

### Florida Models

**Training Data**: Florida voter registration database (early 2017)
**Input**: Last name only OR first + last name
**Categories**: 4 main categories (white, black, asian, hispanic)

```python
from ethnicolr2 import pred_fl_last_name, pred_fl_full_name

# Last name only
ln_predictions = pred_fl_last_name(df, 'last_name')

# First + Last name (highest accuracy)
full_predictions = pred_fl_full_name(df, 'last_name', 'first_name')
```

**Use Cases**:
- Highest accuracy predictions
- When you have both first and last names
- State-specific analysis (trained on Florida data)
- Modern demographic analysis

## Model Architecture

All prediction models use **LSTM (Long Short-Term Memory)** neural networks:

### Character-Level Processing

Names are processed character by character:
```
"Zhang" → ['Z', 'h', 'a', 'n', 'g'] → LSTM → [probabilities]
```

### Model Specifications

| Model | Max Length | Hidden Size | Layers | Vocabulary |
|-------|------------|-------------|--------|------------|
| Census Last Name | 15 chars | 256 units | 2 | Census surnames |
| Florida Last Name | 30 chars | 256 units | 2 | FL voter surnames |  
| Florida Full Name | 47 chars | 256 units | 2 | FL voter full names |

## Prediction Categories

### Standard 5-Category System

Used by census and some Florida models:

- **nh_white**: Non-Hispanic White
- **nh_black**: Non-Hispanic Black
- **hispanic**: Hispanic (any race)
- **asian**: Asian and Pacific Islander
- **other**: Other races/ethnicities

### 4-Category Florida System  

Used by main Florida models:

- **nh_white**: Non-Hispanic White  
- **nh_black**: Non-Hispanic Black
- **hispanic**: Hispanic
- **asian**: Asian

## Understanding Predictions

### Probability Scores

Each prediction includes confidence scores:

```python
result = pred_fl_last_name(df, 'last_name')
print(result.columns)
# ['last_name', 'race', 'asian', 'hispanic', 'nh_black', 'nh_white']

# 'race' = category with highest probability
# Individual columns = probability scores (0-1)
```

### Interpretation Guidelines

**High Confidence** (>0.8 for top category):
```python
# Very confident prediction
{'race': 'asian', 'asian': 0.95, 'hispanic': 0.02, 'nh_black': 0.01, 'nh_white': 0.02}
```

**Medium Confidence** (0.5-0.8):
```python
# Moderately confident
{'race': 'nh_white', 'asian': 0.15, 'hispanic': 0.10, 'nh_black': 0.05, 'nh_white': 0.70}
```

**Low Confidence** (<0.5):
```python
# Uncertain prediction - use with caution
{'race': 'hispanic', 'asian': 0.25, 'hispanic': 0.40, 'nh_black': 0.20, 'nh_white': 0.15}
```

## Data Quality and Limitations

### Model Training Biases

**Census Data**:
- Based on self-reported census responses
- May not capture mixed-race individuals well
- Historical data may not reflect current demographics

**Florida Voter Data**:
- Specific to Florida population
- May not generalize to other states/countries
- Voter registration may skew certain demographics

### Name Variations

Models handle common variations but may struggle with:
- **Spelling variations**: "Smith" vs "Smyth"  
- **Hyphenated names**: "Garcia-Rodriguez"
- **Non-Western names**: Names from underrepresented populations
- **Nicknames**: "Bob" vs "Robert"

### Best Practices

1. **Use appropriate model for your use case**:
   - Census models for academic/historical analysis
   - Florida models for highest accuracy

2. **Consider confidence scores**:
   - Don't use predictions with very low confidence
   - Flag uncertain predictions for manual review

3. **Validate with known data**:
   - Test accuracy on a subset with known ethnicity
   - Compare results across different models

4. **Be aware of ethical considerations**:
   - Use predictions responsibly
   - Consider privacy and bias implications
   - Don't use for discriminatory purposes

## Preprocessing and Data Flow

### Input Processing

1. **Name normalization**: Convert to lowercase, remove special characters
2. **Character encoding**: Convert to numerical sequences  
3. **Padding**: Ensure uniform length for batch processing
4. **Tensor conversion**: Convert to PyTorch tensors

### Model Inference

1. **Embedding**: Characters → dense vectors
2. **LSTM layers**: Process sequences → hidden states
3. **Linear layer**: Hidden states → class probabilities
4. **Softmax**: Convert to probability distribution

### Output Processing

1. **Probability extraction**: Get scores for each category
2. **Category assignment**: Select highest probability 
3. **DataFrame integration**: Merge with original data
4. **Result formatting**: Clean column names and types

## Next Steps

- {doc}`../user-guide/census-data`: Detailed census model usage
- {doc}`../user-guide/florida-models`: Florida model deep dive  
- {doc}`../user-guide/examples`: Practical examples and case studies
- {doc}`../api-reference/models`: Technical API documentation