# Basic Usage Guide

This guide demonstrates the fundamentals of ethnicolr2 with step-by-step examples using all available prediction models.

## Overview

ethnicolr2 provides three main prediction models:
- **Census Last Name**: Predicts from last names using US Census data
- **Florida Last Name**: Predicts from last names using Florida voter registration data
- **Florida Full Name**: Predicts using both first and last names for highest accuracy

Let's start by importing the necessary libraries and loading sample data.


```python
import pandas as pd
from pathlib import Path
import sys

# Import ethnicolr2 prediction functions
from ethnicolr2 import (
    census_ln,
    pred_fl_last_name,
    pred_fl_full_name,
    pred_census_last_name
)

print(f"Python version: {sys.version}")
print(f"pandas version: {pd.__version__}")
print("ethnicolr2 imported successfully!")
```

## Sample Data

Let's create a sample dataset with diverse names to demonstrate the prediction capabilities:


```python
# Create sample data with diverse names
sample_data = {
    'first_name': ['John', 'Maria', 'Wei', 'Aisha', 'David', 'Priya', 'Carlos', 'Sarah'],
    'last_name': ['Smith', 'Rodriguez', 'Zhang', 'Johnson', 'Williams', 'Patel', 'Garcia', 'Kim'],
    'full_name': ['John Smith', 'Maria Rodriguez', 'Wei Zhang', 'Aisha Johnson',
                  'David Williams', 'Priya Patel', 'Carlos Garcia', 'Sarah Kim']
}

df = pd.DataFrame(sample_data)
print("Sample dataset:")
display(df)
```

## 1. Census Last Name Predictions

The census model uses US Census data for predictions based on last names only:


```python
# Census last name predictions
census_results = pred_census_last_name(df.copy(), lname_col='last_name')

print("Census Last Name Model Results:")
print("=" * 50)

# Display results with predictions and probabilities
display_cols = ['first_name', 'last_name', 'preds']
display(census_results[display_cols])

# Show probability distribution for first few rows
print("\nProbability distributions (first 3 rows):")
for i, (idx, row) in enumerate(census_results.head(3).iterrows()):
    print(f"{row['last_name']}: {row['probs']}")
```

## 2. Florida Last Name Predictions

The Florida model is trained on Florida voter registration data:


```python
# Florida last name predictions
fl_ln_results = pred_fl_last_name(df.copy(), lname_col='last_name')

print("Florida Last Name Model Results:")
print("=" * 50)

# Display results
display(fl_ln_results[display_cols])

# Show probability distribution for first few rows
print("\nProbability distributions (first 3 rows):")
for i, (idx, row) in enumerate(fl_ln_results.head(3).iterrows()):
    print(f"{row['last_name']}: {row['probs']}")
```

## 3. Florida Full Name Predictions

The most accurate model uses both first and last names:


```python
# Florida full name predictions using first and last name columns
fl_full_results = pred_fl_full_name(df.copy(),
                                   fname_col='first_name',
                                   lname_col='last_name')

print("Florida Full Name Model Results:")
print("=" * 50)

# Display results
display(fl_full_results[display_cols])

# Show probability distribution for first few rows
print("\nProbability distributions (first 3 rows):")
for i, (idx, row) in enumerate(fl_full_results.head(3).iterrows()):
    print(f"{row['first_name']} {row['last_name']}: {row['probs']}")
```

## Summary

This guide demonstrated:

1. **Three prediction models** with different data sources and accuracy levels
2. **Easy Python API** for batch predictions on DataFrames
3. **Probability distributions** for uncertainty quantification
4. **Model comparison** to understand prediction differences

### Key Takeaways

- **Florida Full Name** model generally provides the most accurate predictions
- **Probability distributions** are valuable for understanding prediction confidence
- **Different models** may disagree, especially for ambiguous names
- **Sample data** works seamlessly without external dependencies
