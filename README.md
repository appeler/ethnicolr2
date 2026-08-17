# ethnicolr2: Predict Race and Ethnicity From Name

[![image](https://github.com/appeler/ethnicolr2/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/ethnicolr2/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/ethnicolr2.svg)](https://pypi.org/project/ethnicolr2)
[![Python version](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/appeler/ethnicolr2/main/pyproject.toml&query=$.project.requires-python&label=Python&color=green)](https://github.com/appeler/ethnicolr2)
[![image](https://static.pepy.tech/badge/ethnicolr2)](https://pepy.tech/project/ethnicolr2)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://appeler.github.io/ethnicolr2/)

## Project status

`ethnicolr2` is in maintenance mode. Existing users can keep using it, and we
will continue to fix serious bugs, security issues, and compatibility breaks.
New projects should use [ethnicolr](https://github.com/appeler/ethnicolr), the
canonical package. New models and features will be developed there.

`ethnicolr2` preserves three PyTorch LSTM models trained on US Census and
Florida voter registration data. The models predict five race and ethnicity
categories from a last name or from a first and last name.

# Caveats and Notes

For a random person named Smith in the 2010 US Census population, the modal
race among people named Smith is the Bayes-optimal point prediction. A model is
most useful when a name is missing from the Census table or when both first and
last names are available. Predictions outside the model's training population
require assumptions that may not hold.

# Install

Install `ethnicolr2` inside a Python virtual environment (see the [venv
documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments)).

    pip install ethnicolr2

# Example

To predict race/ethnicity using the Florida Last Name Model to a [file
with first and last names](docs/source/examples/input-with-header.csv)

    import pandas as pd
    from ethnicolr2 import pred_fl_last_name, pred_fl_full_name
    df = pd.read_csv("docs/source/examples/input-with-header.csv")
    pred_fl_last_name(df, lname_col = "last_name")


    names = [
     {"last": "sawyer", "first": "john", "true_race": "nh_white"},
     {"last": "torres", "first": "raul", "true_race": "hispanic"},
    ]
    df = pd.DataFrame(names)
    df = pred_fl_full_name(df, lname_col = "last", fname_col = "first")

          last  first true_race   preds
    0  sawyer   john  nh_white nh_white
    1  torres   raul  hispanic hispanic

# Authors

Rajashekar Chintalapati, Suriyan Laohaprapanon, and Gaurav Sood

# Contributor Code of Conduct

The project welcomes contributions from everyone. To maintain a welcoming
atmosphere, contributors must abide by the [Contributor Code of
Conduct](http://contributor-covenant.org/version/1/0/0/).
