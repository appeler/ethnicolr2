# Utilities API Reference

This page documents utility functions and command-line tools.

## Command-Line Interface

ethnicolr2 provides several command-line tools for batch processing:

### Census Data Lookup

```bash
# Look up census data by last name
census_ln input.csv -l last_name -o output.csv
```

### Prediction Commands

```bash
# Florida last name model
pred_fl_last_name input.csv -l last_name -o predictions.csv

# Florida full name model
pred_fl_full_name input.csv -l last_name -f first_name -o predictions.csv

# Census last name model
pred_census_last_name input.csv -l last_name -o predictions.csv
```

### Model Download

```bash
# Download pre-trained models (if needed)
ethnicolr2_download_models
```

## Command Line Options

All command-line tools support these common options:

- `-h, --help`: Show help message and exit
- `-o OUTPUT, --output OUTPUT`: Output file path
- `-l LAST, --last LAST`: Column name or index for last names
- `-f FIRST, --first FIRST`: Column name or index for first names (where applicable)

## Programmatic Usage

You can also use the command-line functionality programmatically:

```python
import subprocess
import pandas as pd

# Prepare input data
df = pd.DataFrame({'surname': ['Smith', 'Zhang', 'Rodriguez']})
df.to_csv('input.csv', index=False)

# Run prediction via command line
result = subprocess.run([
    'pred_fl_last_name',
    'input.csv',
    '-l', 'surname',
    '-o', 'output.csv'
], capture_output=True, text=True)

# Load results
if result.returncode == 0:
    predictions = pd.read_csv('output.csv')
    print(predictions)
else:
    print(f"Error: {result.stderr}")
```

## Batch Processing

For large datasets, the command-line interface automatically handles batch processing:

```bash
# Process large CSV files efficiently
pred_fl_last_name large_dataset.csv -l lastname -o results.csv
```

The tools automatically:
- Process data in chunks to manage memory usage
- Show progress bars for long-running operations
- Handle encoding issues gracefully
- Preserve original column order and additional columns
