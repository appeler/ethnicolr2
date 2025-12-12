# Example Notebooks

This directory contains Jupyter notebooks demonstrating the use of ethnicolr2 for various applications.

## External Dependencies

These example notebooks use some external packages that are not part of the main ethnicolr2 package dependencies:

- **`clean_names`**: Used for name cleaning/standardization
- **`ethnicolr`**: The original ethnicolr package (legacy) - used for comparison

## Installation

To run these examples, install the required external packages:

```bash
pip install clean_names ethnicolr
```

## Notebooks

- `ethnicolr_app_contrib2000.ipynb`: Campaign contributor analysis for 2000 data
- `ethnicolr_app_contrib2010.ipynb`: Campaign contributor analysis for 2010 data
- `ethnicolr_app_contrib20xx-census_ln.ipynb`: Census last name analysis
- `ethnicolr_app_contrib20xx-fl_reg.ipynb`: Florida registration analysis
- `ethnicolr_app_contrib20xx.ipynb`: General contributor analysis

These notebooks demonstrate real-world applications and provide comparison with the legacy ethnicolr package.
