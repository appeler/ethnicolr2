# ethnicolr2: Predict Race and Ethnicity From Name

[![image](https://github.com/appeler/ethnicolr2/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/ethnicolr2/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/ethnicolr2.svg)](https://pypi.org/project/ethnicolr2)
[![Python version](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/appeler/ethnicolr2/main/pyproject.toml&query=$.project.requires-python&label=Python&color=green)](https://github.com/appeler/ethnicolr2)
[![image](https://static.pepy.tech/badge/ethnicolr2)](https://pepy.tech/project/ethnicolr2)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://appeler.github.io/ethnicolr2/)

A Pytorch implementation of
[ethnicolr](https://github.com/appeler/ethnicolr) with new models that
make different assumptions (for instance, this package has models
trained on unique names) than ethnicolr. The package uses the US census
data and the Florida voting registration data to build models to predict
the race and ethnicity (non-Hispanic whites, non-Hispanic Blacks,
Asians, Hispanics, and Other) based on first and last name or just the
last name. For notebooks underlying the package, see
[here](https://github.com/appeler/ethnicolr_v2).

# Caveats and Notes

If you picked a random person with the last name \'Smith\' in the US in
2010 and asked us to guess this person\'s race (as measured by the
census), the best guess is the modal race of the person named Smith
(which you can get from the census popular last name data file). It is
the Bayes Optimal Solution. So what good are predictive models? A few
things\-\--if you want to impute race and ethnicity for last names that
are not in the census file, which can be because of errors, infer the
race and ethnicity in different years than when the census was conducted
(if some assumptions hold), infer the race of people in different
countries (if some assumptions hold), etc. The biggest benefit comes in
cases where both the first and last name are known.

# Install

We strongly recommend installing [ethnicolor2]{.title-ref} inside a
Python virtual environment (see [venv
documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

    pip install ethnicolr2

# Example

To predict race/ethnicity using the Florida Last Name Model to a [file
with first and last names](ethnicolr2/data/input-with-header.csv)

    import pandas as pd
    from ethnicolr2 import pred_fl_last_name, pred_fl_full_name
    df = pd.read_csv("ethnicolr2/data/input-with-header.csv")
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

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the [Contributor Code of
Conduct](http://contributor-covenant.org/version/1/0/0/).


## 🔗 Adjacent Repositories

- [appeler/ethnicolr](https://github.com/appeler/ethnicolr) — Predict Race and Ethnicity Based on the Sequence of Characters in a Name
- [appeler/ethnicolor](https://github.com/appeler/ethnicolor) — Race and Ethnicity based on name using data from census, voter reg. files, etc.
- [appeler/parsernaam](https://github.com/appeler/parsernaam) — AI name parsing. Predict first or last name using a DL model.
- [appeler/instate](https://github.com/appeler/instate) — instate: predict the state of residence from last name using the indian electoral rolls
- [appeler/naamkaran](https://github.com/appeler/naamkaran) — generative model for names
