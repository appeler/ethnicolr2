# Census Data Models

The census data models in ethnicolr2 provide predictions based on US Census surname statistics and LSTM models trained on census data.

## Available Census Functions

### census_ln() - Census Statistics Lookup

Direct lookup of census surname statistics without machine learning prediction:

```python
from ethnicolr2 import census_ln
import pandas as pd

# Create sample data
df = pd.DataFrame({'surname': ['Smith', 'Zhang', 'Rodriguez', 'Johnson']})

# Get 2010 census statistics
census_2010 = census_ln(df, 'surname', year=2010)
print(census_2010)
```

**Output columns**:
- `pctwhite`: Percentage Non-Hispanic White
- `pctblack`: Percentage Non-Hispanic Black  
- `pctapi`: Percentage Asian and Pacific Islander
- `pctaian`: Percentage American Indian and Alaska Native
- `pct2prace`: Percentage Two or More Races
- `pcthispanic`: Percentage Hispanic

### pred_census_last_name() - LSTM Predictions

Machine learning predictions using LSTM models trained on census data:

```python
from ethnicolr2 import pred_census_last_name

# LSTM-based predictions
ml_predictions = pred_census_last_name(df, 'surname', year=2010)
print(ml_predictions)
```

**Output columns**:
- `race`: Predicted category (highest probability)
- `asian`: Probability Asian
- `black`: Probability Non-Hispanic Black
- `hispanic`: Probability Hispanic  
- `white`: Probability Non-Hispanic White

## Census Years

Both 2000 and 2010 census data are available:

```python
# Compare across census years
census_2000 = census_ln(df, 'surname', year=2000)
census_2010 = census_ln(df, 'surname', year=2010)

# ML predictions for different years
pred_2000 = pred_census_last_name(df, 'surname', year=2000)
pred_2010 = pred_census_last_name(df, 'surname', year=2010)
```

## Practical Examples

### Academic Research

```python
import pandas as pd
from ethnicolr2 import census_ln, pred_census_last_name

# Load research dataset
authors_df = pd.read_csv('academic_authors.csv')  
# Columns: ['author_name', 'last_name', 'institution', 'field']

# Get census statistics
census_stats = census_ln(authors_df, 'last_name', year=2010)

# Add ML predictions
ml_predictions = pred_census_last_name(authors_df, 'last_name', year=2010)

# Merge results
research_results = pd.merge(
    authors_df, 
    census_stats[['last_name', 'pctwhite', 'pctblack', 'pctapi', 'pcthispanic']], 
    on='last_name'
)
research_results = pd.merge(
    research_results,
    ml_predictions[['last_name', 'race', 'asian', 'black', 'hispanic', 'white']],
    on='last_name'
)

print(research_results.groupby(['field', 'race']).size())
```

### Historical Analysis

```python
# Compare demographic trends over time
names = ['Kim', 'Patel', 'Martinez', 'Johnson']
df = pd.DataFrame({'last_name': names})

# Get both census years
results_2000 = census_ln(df, 'last_name', year=2000)
results_2010 = census_ln(df, 'last_name', year=2010)

# Compare changes
for name in names:
    row_2000 = results_2000[results_2000['last_name'] == name].iloc[0]
    row_2010 = results_2010[results_2010['last_name'] == name].iloc[0]
    
    print(f"\\n{name}:")
    print(f"  Hispanic 2000: {row_2000['pcthispanic']:.1f}%")
    print(f"  Hispanic 2010: {row_2010['pcthispanic']:.1f}%")
    print(f"  Change: {row_2010['pcthispanic'] - row_2000['pcthispanic']:+.1f}%")
```

### Large Dataset Processing

```python
def process_census_batch(df_chunk, year=2010):
    \"\"\"Process large datasets in batches\"\"\"
    # Get both census stats and ML predictions
    census_stats = census_ln(df_chunk, 'last_name', year=year)
    ml_predictions = pred_census_last_name(df_chunk, 'last_name', year=year)
    
    # Merge results
    result = pd.merge(df_chunk, census_stats, on='last_name')
    result = pd.merge(result, ml_predictions[['last_name', 'race']], on='last_name')
    
    return result

# Process large CSV in chunks
chunk_size = 10000
results = []

for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
    chunk_result = process_census_batch(chunk)
    results.append(chunk_result)
    print(f"Processed {len(chunk)} records")

final_result = pd.concat(results, ignore_index=True)
```

## Data Quality Considerations

### Coverage

Not all surnames appear in census data:

```python
# Check which names have census data
census_result = census_ln(df, 'last_name', year=2010)

# Names not in census will have NaN values
missing_census = census_result[census_result['pctwhite'].isna()]
print(f"Names missing from census: {len(missing_census)}")

# ML predictions work for all names (including those not in census)
ml_result = pred_census_last_name(df, 'last_name', year=2010)
print(f"ML predictions available: {len(ml_result)}")
```

### Confidence Assessment

```python
# Assess prediction confidence
ml_predictions = pred_census_last_name(df, 'last_name', year=2010)

# Calculate max probability (confidence indicator)
ml_predictions['confidence'] = ml_predictions[['asian', 'black', 'hispanic', 'white']].max(axis=1)

# High confidence predictions (>80%)
high_confidence = ml_predictions[ml_predictions['confidence'] > 0.8]
print(f"High confidence predictions: {len(high_confidence)} / {len(ml_predictions)}")

# Review uncertain predictions
uncertain = ml_predictions[ml_predictions['confidence'] < 0.5]
print("\\nUncertain predictions:")
print(uncertain[['last_name', 'race', 'confidence']])
```

## Census vs Florida Models

When to use census models vs Florida models:

### Use Census Models When:
- Academic research requiring census validation
- Historical analysis (2000 vs 2010 comparison)
- Need to match published census statistics
- Working with surnames only
- Analyzing general US population

### Use Florida Models When:
- Need highest accuracy
- Have both first and last names available
- Working with modern datasets
- Doing practical applications (not academic)

```python
# Compare census vs Florida predictions
from ethnicolr2 import pred_census_last_name, pred_fl_last_name

census_pred = pred_census_last_name(df, 'last_name')
florida_pred = pred_fl_last_name(df, 'last_name')

# Compare predictions
comparison = pd.DataFrame({
    'name': df['last_name'],
    'census_race': census_pred['race'],
    'florida_race': florida_pred['race'],
    'census_conf': census_pred[['asian', 'black', 'hispanic', 'white']].max(axis=1),
    'florida_conf': florida_pred[['asian', 'hispanic', 'nh_black', 'nh_white']].max(axis=1)
})

# Check agreement
agreement = (comparison['census_race'] == comparison['florida_race']).mean()
print(f"Census vs Florida agreement: {agreement:.2%}")
```

## Command Line Usage

```bash
# Census statistics lookup
census_ln input.csv -l last_name -o census_output.csv -y 2010

# Census LSTM predictions  
pred_census_last_name input.csv -l last_name -o ml_output.csv -y 2010
```

## Next Steps

- {doc}`florida-models`: Explore higher-accuracy Florida models
- {doc}`examples`: See more practical examples
- {doc}`../api-reference/models`: Technical API documentation