## Census Last Name Data

The Census Bureau provides frequency of all surnames occurring 100 or more times for the [2000](https://www.census.gov/topics/population/genealogy/data/2000_surnames.html) and [2010](https://www.census.gov/topics/population/genealogy/data/2010_surnames.html) census. Technical details of how the 2000 and 2010 data were collected can be found in [the 2000 documentation](census_2000.pdf) and [the 2010 documentation](census_2010.pdf).

In the Census data, percentages based on one to four people are suppressed and replaced with `(S)`. [The table builder](../../build_census_tables.py) divides each row's remaining percentage equally across its suppressed cells and writes the typed runtime tables under `src/ethnicolr2/data/census`.
