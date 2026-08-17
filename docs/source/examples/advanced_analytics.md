# Advanced Analytics and Performance

This guide explores advanced features of ethnicolr2 including batch processing, performance optimization, and statistical analysis.


```python
import pandas as pd
import numpy as np
import time

# Import ethnicolr2 functions
from ethnicolr2 import (
    pred_fl_last_name,
    pred_fl_full_name,
    pred_census_last_name
)

print("Advanced analytics setup complete")
```

## Performance Benchmarking

Let's test performance with a larger synthetic dataset:


```python
# Create a larger synthetic dataset
np.random.seed(42)

# Common surnames from different ethnic backgrounds
surnames = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones',  # Common American
    'Garcia', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez',  # Hispanic
    'Zhang', 'Wang', 'Li', 'Liu', 'Chen',  # Chinese
    'Kim', 'Lee', 'Park', 'Choi', 'Jung',  # Korean
    'Patel', 'Shah', 'Singh', 'Kumar', 'Sharma'  # South Asian
]

# Generate synthetic dataset
n_samples = 1000
large_df = pd.DataFrame({
    'id': range(n_samples),
    'last_name': np.random.choice(surnames, n_samples)
})

print(f"Created dataset with {len(large_df)} rows")
print(f"Unique surnames: {large_df['last_name'].nunique()}")
display(large_df.head())
```


```python
# Benchmark different models
models = {
    'Census Last Name': lambda df: pred_census_last_name(df, lname_col='last_name'),
    'Florida Last Name': lambda df: pred_fl_last_name(df, lname_col='last_name')
}

performance_results = []

for model_name, model_func in models.items():
    print(f"Benchmarking {model_name}...")

    start_time = time.time()
    result_df = model_func(large_df.copy())
    end_time = time.time()

    execution_time = end_time - start_time
    rows_per_second = len(large_df) / execution_time

    performance_results.append({
        'Model': model_name,
        'Rows': len(large_df),
        'Time (s)': round(execution_time, 3),
        'Rows/sec': round(rows_per_second, 1)
    })

    print(f"  Time: {execution_time:.3f}s ({rows_per_second:.1f} rows/sec)")

perf_df = pd.DataFrame(performance_results)
print("\nPerformance Summary:")
display(perf_df)
```

## Statistical Analysis

Analyze prediction confidence and uncertainty:


```python
# Analyze prediction confidence
results = pred_fl_last_name(large_df.copy(), lname_col='last_name')

# Extract max probabilities (confidence)
max_probs = []
for _, row in results.iterrows():
    probs = row['probs']
    max_prob = max(probs.values())
    max_probs.append(max_prob)

results['confidence'] = max_probs

print("Confidence Statistics:")
print(f"Mean confidence: {np.mean(max_probs):.3f}")
print(f"Median confidence: {np.median(max_probs):.3f}")
print(f"Min confidence: {np.min(max_probs):.3f}")
print(f"Max confidence: {np.max(max_probs):.3f}")

# Confidence by prediction category
confidence_by_pred = results.groupby('preds')['confidence'].agg(['mean', 'count'])
confidence_by_pred.columns = ['avg_confidence', 'count']
confidence_by_pred = confidence_by_pred.round(3)

print("\nConfidence by Prediction Category:")
display(confidence_by_pred)
```

## Summary

This advanced analytics guide demonstrated:

1. **Performance benchmarking** across different models
2. **Statistical analysis** of prediction confidence
3. **Large-scale processing** patterns

### Key Insights

- ethnicolr2 efficiently handles large datasets
- Probability distributions provide valuable uncertainty information
- Different models have different performance characteristics
- Confidence scores help identify uncertain predictions
