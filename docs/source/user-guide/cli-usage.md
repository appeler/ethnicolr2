# Command Line Interface

ethnicolr2 provides command-line tools for batch processing of CSV files, making it easy to integrate into data processing pipelines.

## Available Commands

### census_ln - Census Statistics Lookup

Get census surname statistics without machine learning:

```bash
census_ln input.csv -l last_name -o output.csv -y 2010
```

**Options:**
- `-l, --last`: Column name or index containing last names
- `-o, --output`: Output CSV filename  
- `-y, --year`: Census year (2000 or 2010, default: 2000)

### pred_census_last_name - Census LSTM Predictions

Machine learning predictions using census-trained models:

```bash
pred_census_last_name input.csv -l surname -o predictions.csv -y 2010
```

### pred_fl_last_name - Florida Last Name Model

High-accuracy predictions using Florida voter data:

```bash
pred_fl_last_name input.csv -l last_name -o fl_predictions.csv
```

### pred_fl_full_name - Florida Full Name Model  

Highest accuracy using both first and last names:

```bash
pred_fl_full_name input.csv -l last_name -f first_name -o full_predictions.csv
```

**Options:**
- `-l, --last`: Last name column
- `-f, --first`: First name column
- `-o, --output`: Output filename

## Input File Formats

### With Headers

```csv
first_name,last_name,employee_id
John,Smith,12345
Maria,Rodriguez,12346
Wei,Zhang,12347
```

```bash
# Use column names
pred_fl_full_name employees.csv -l last_name -f first_name -o results.csv
```

### Without Headers

```csv
John,Smith,12345
Maria,Rodriguez,12346  
Wei,Zhang,12347
```

```bash
# Use column indices (0-based)
pred_fl_full_name employees.csv -l 1 -f 0 -o results.csv
```

## Practical Examples

### Process Employee Database

```bash
# Input: employees.csv
# Columns: emp_id,first_name,last_name,department,salary

# Get demographic predictions for HR analysis
pred_fl_full_name employees.csv \\
  --last last_name \\
  --first first_name \\
  --output employee_demographics.csv

# Results include original data + predictions
head employee_demographics.csv
```

### Academic Research Dataset

```bash
# Input: research_authors.csv  
# Columns: paper_id,author_surname,institution,field

# Use census model for academic validation
pred_census_last_name research_authors.csv \\
  --last author_surname \\
  --output author_demographics.csv \\
  --year 2010
```

### Customer Analysis Pipeline

```bash
#!/bin/bash
# Pipeline for customer demographic analysis

# Step 1: Extract customer names
cut -d',' -f2,3 customers_full.csv > customer_names.csv

# Step 2: Get demographic predictions
pred_fl_last_name customer_names.csv \\
  -l 1 \\
  -o customer_demographics.csv

# Step 3: Merge back with original data
python merge_results.py customers_full.csv customer_demographics.csv
```

### Batch Processing Multiple Files

```bash
#!/bin/bash
# Process multiple CSV files

for file in data/*.csv; do
    echo "Processing $file..."
    pred_fl_last_name "$file" \\
      -l last_name \\
      -o "results/$(basename "$file" .csv)_demographics.csv"
done
```

## Performance Tips

### Large Files

For files larger than 100MB, consider splitting:

```bash
# Split large file into chunks
split -l 50000 large_dataset.csv chunk_

# Process chunks in parallel
for chunk in chunk_*; do
    pred_fl_last_name "$chunk" -l 1 -o "results_$chunk.csv" &
done
wait

# Combine results
cat results_chunk_*.csv > final_results.csv
```

### Memory Usage

Monitor memory usage for very large datasets:

```bash
# Check available memory
free -h

# Run with memory monitoring
/usr/bin/time -v pred_fl_last_name large_file.csv -l last_name -o output.csv
```

## Error Handling

### Common Issues

**FileNotFoundError:**
```bash
# Check file exists
ls -la input.csv

# Use absolute path if needed
pred_fl_last_name /full/path/to/input.csv -l last_name -o output.csv
```

**Column Not Found:**
```bash
# Check column names
head -1 input.csv

# Use correct column name or index
pred_fl_last_name input.csv -l "Last Name" -o output.csv  # with spaces
pred_fl_last_name input.csv -l 2 -o output.csv           # by index
```

**Permission Errors:**
```bash
# Check write permissions
ls -la output_directory/

# Use different output location
pred_fl_last_name input.csv -l last_name -o ~/Desktop/output.csv
```

### Validation

Verify results before using:

```bash
# Check output file structure
head output.csv
wc -l input.csv output.csv  # Should have same number of lines

# Quick statistics
cut -d',' -f'race_column' output.csv | sort | uniq -c
```

## Integration with Data Pipelines

### Apache Airflow

```python
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

dag = DAG('demographic_analysis', 
          start_date=datetime(2023, 1, 1),
          schedule_interval='@daily')

predict_demographics = BashOperator(
    task_id='predict_demographics',
    bash_command='''
    pred_fl_last_name /data/daily_customers.csv \\
      -l last_name \\
      -o /data/demographics_{{ ds }}.csv
    ''',
    dag=dag
)
```

### Make/Unix Pipelines

```makefile
# Makefile for demographic analysis

demographics.csv: raw_data.csv
\tpred_fl_last_name $< -l last_name -o $@

analysis.html: demographics.csv
\tpython generate_report.py $< > $@

clean:
\trm -f demographics.csv analysis.html
```

### Docker Integration

```dockerfile
FROM python:3.11-slim

RUN pip install ethnicolr2

WORKDIR /app
COPY process.sh /app/

ENTRYPOINT ["./process.sh"]
```

```bash
# Run in Docker
docker run -v $(pwd)/data:/app/data demographic-processor \\
  pred_fl_last_name /app/data/input.csv -l last_name -o /app/data/output.csv
```

## Output Customization

### Selecting Columns

Most CLI tools output all original columns plus predictions. To select specific columns:

```bash
# Process then select columns
pred_fl_last_name input.csv -l last_name -o full_output.csv
cut -d',' -f1,2,race,asian,hispanic full_output.csv > selected_output.csv
```

### Formatting

```bash
# Convert to TSV
pred_fl_last_name input.csv -l last_name -o temp.csv
tr ',' '\\t' < temp.csv > output.tsv

# Add headers if needed
echo -e "name\\trace\\tconfidence" > final.tsv
tail -n +2 output.tsv >> final.tsv
```

## Getting Help

```bash
# Get help for any command
census_ln --help
pred_fl_last_name --help
pred_fl_full_name --help

# Check version
python -c "import ethnicolr2; print(ethnicolr2.__version__)"
```

## Next Steps

- {doc}`examples`: See complete workflow examples
- {doc}`../api-reference/utilities`: Python API for more control
- {doc}`troubleshooting`: Common issues and solutions