# 5-Minute Quickstart

This quickstart guide will get you up and running with ethnicolr2 in just a few minutes.

## Basic Example

```python
import pandas as pd
from ethnicolr2 import pred_fl_last_name

# Create sample data
names_df = pd.DataFrame({
    'last_name': ['Smith', 'Zhang', 'Rodriguez', 'Johnson', 'Kim']
})

# Predict race/ethnicity from last names
results = pred_fl_last_name(names_df, lname_col='last_name')
print(results)
```

**Output:**
```
  last_name      race     asian   hispanic   nh_black   nh_white
0     Smith  nh_white  0.001234   0.012345   0.234567   0.751854
1     Zhang     asian  0.987654   0.001234   0.002345   0.008767  
2 Rodriguez  hispanic  0.001234   0.934567   0.012345   0.051854
3   Johnson  nh_white  0.001234   0.023456   0.123456   0.851854
4       Kim     asian  0.876543   0.012345   0.023456   0.087656
```

## Different Models

### Census Data Model

For general population predictions based on US Census data:

```python
from ethnicolr2 import census_ln

# Get census statistics for names
census_results = census_ln(names_df, 'last_name')
print(census_results.columns)
# ['last_name', 'pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']
```

### Full Name Model (Highest Accuracy)

When you have both first and last names:

```python
from ethnicolr2 import pred_fl_full_name

# Create data with first and last names
full_names_df = pd.DataFrame({
    'first_name': ['John', 'Wei', 'Maria', 'Robert', 'Priya'],
    'last_name': ['Smith', 'Zhang', 'Rodriguez', 'Johnson', 'Patel']
})

# Predict using both names (most accurate)
full_results = pred_fl_full_name(
    full_names_df, 
    lname_col='last_name', 
    fname_col='first_name'
)
print(full_results)
```

## Command Line Usage

ethnicolr2 also provides command-line tools:

```bash
# Census lookup
census_ln input.csv -l last_name -o output.csv

# Florida last name prediction  
pred_fl_last_name input.csv -l last_name -o output.csv

# Florida full name prediction
pred_fl_full_name input.csv -l last_name -f first_name -o output.csv
```

## Understanding the Output

Each model returns probability scores for different racial/ethnic categories:

- **race**: The predicted category (highest probability)
- **asian**: Probability of Asian ethnicity
- **hispanic**: Probability of Hispanic ethnicity  
- **nh_black**: Probability of Non-Hispanic Black
- **nh_white**: Probability of Non-Hispanic White
- **other**: Probability of Other (in some models)

## Input Data Requirements

### Pandas DataFrame

```python
# Your data must be a pandas DataFrame
df = pd.DataFrame({'names': ['Smith', 'Zhang']})

# Specify which column contains the names
result = pred_fl_last_name(df, lname_col='names')
```

### CSV Files

```csv
first_name,last_name,id
John,Smith,1
Wei,Zhang,2
Maria,Rodriguez,3
```

```python
import pandas as pd

# Read CSV file
df = pd.read_csv('names.csv')
result = pred_fl_full_name(df, lname_col='last_name', fname_col='first_name')
```

## Handling Missing Data

```python
# DataFrame with missing values
df = pd.DataFrame({
    'last_name': ['Smith', None, 'Zhang', ''],
    'first_name': ['John', 'Maria', 'Wei', None]
})

# Missing values are handled automatically
# Empty strings and None values will receive default predictions
result = pred_fl_last_name(df, lname_col='last_name')
```

## Performance Tips

```python
# For large datasets, process in chunks
import pandas as pd

def process_large_dataset(df, chunk_size=1000):
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df[i:i+chunk_size]
        chunk_result = pred_fl_last_name(chunk, lname_col='last_name')
        results.append(chunk_result)
    return pd.concat(results, ignore_index=True)

# Process 100K records efficiently
large_df = pd.read_csv('large_dataset.csv')
results = process_large_dataset(large_df)
```

## What's Next?

- {doc}`concepts`: Learn about the different models and datasets
- {doc}`../user-guide/census-data`: Deep dive into census predictions
- {doc}`../user-guide/florida-models`: Explore Florida voter models
- {doc}`../api-reference/index`: Complete API documentation