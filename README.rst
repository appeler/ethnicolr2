ethnicolr2: Predict Race and Ethnicity From Name
----------------------------------------------------

.. image:: https://github.com/appeler/ethnicolr2/workflows/test/badge.svg
    :target: https://github.com/appeler/ethnicolr2/actions?query=workflow%3Atest
.. image:: https://img.shields.io/pypi/v/ethnicolr2.svg
    :target: https://pypi.python.org/pypi/ethnicolr2
.. image:: https://pepy.tech/badge/ethnicolr2
    :target: https://pepy.tech/project/ethnicolr2

A pytorch implementation of `ethnicolr <https://github.com/appeler/ethnicolr>`__  with new models that make different assumptions (for instance, this package has models trained on unique names) than ethnicolr. The package uses the US census data and the Florida voting registration data to build models to predict the race and ethnicity (Non-Hispanic Whites, Non-Hispanic Blacks, Asians, Hispanics, and Other) based on first and last name or just the last name. For notebooks underlying the package, see `here <https://github.com/appeler/ethnicolr_v2>`__.

Caveats and Notes
-----------------------

If you picked a random person with the last name 'Smith' in the US in 2010 and asked us to guess this person's race (as measured by the census), the best guess is the modal race of the person named Smith (which you can get from the census popular last name data file). It is the Bayes Optimal Solution. So what good are predictive models? A few things---if you want to impute race and ethnicity for last names that are not in the census file, which can be because of errors, infer the race and ethnicity in different years than when the census was conducted (if some assumptions hold), infer the race of people in different countries (if some assumptions hold), etc. The biggest benefit comes in cases where both the first and last name are known.

Install
----------

We strongly recommend installing `ethnicolor2` inside a Python virtual environment
(see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install ethnicolr2

General API
------------------

To see the available command line options for any function, please type in 
``<function-name> --help``

::

   # census_ln --help
   usage: census_ln [-h] [-y {2000,2010}] [-o OUTPUT] -l LAST input

   Appends Census columns by last name

   positional arguments:
     input                 Input file

   optional arguments:
     -h, --help            show this help message and exit
     -y {2000,2010}, --year {2000,2010}
                           Year of Census data (default=2000)
     -o OUTPUT, --output OUTPUT
                           Output file with Census data columns
     -l LAST, --last LAST  Name of the column containing the last name


Examples
----------

To append census data from 2010 to a `file with column header in the first row <ethnicolr2/data/input-with-header.csv>`__, specify the column name carrying last names using the ``-l`` option, keeping the rest the same:

::

   census_ln -y 2010 -o output-census2010.csv -l last_name input-with-header.csv   


To predict race/ethnicity using the Florida Last Name Model, specify the column name of last name and first name by using ``-l`` and ``-f`` flags respectively.

::

   pred_fl_last_name -o output-wiki-pred-race.csv -l last_name -f first_name input-with-header.csv


Functions
----------

We expose 4 functions, each of which either take a pandas DataFrame or a
CSV.

- **census\_ln(df, lname_col, year=2000)**

  -  What it does:

     - Removes extra space
     - For names in the `census file <ethnicolr/data/census>`__, it appends 
       relevant data of what probability the name provided is of a certain race/ethnicity

 +------------+--------------------------------------------------------------------------------------------------------------------------+
 | Parameters |                                                                                                                          |
 +============+==========================================================================================================================+
 |            | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred             |
 +------------+--------------------------------------------------------------------------------------------------------------------------+
 |            | **lname_col** : *{string}* name of the column containing the last name                                                   |
 +------------+--------------------------------------------------------------------------------------------------------------------------+
 |            | **Year** : *{2000, 2010}, default=2000* year of census to use                                                            |
 +------------+--------------------------------------------------------------------------------------------------------------------------+


-  Output: Appends the following columns to the pandas DataFrame or CSV: 
   pctwhite, pctblack, pctapi, pctaian, pct2prace, pcthispanic. 
   See `here <https://github.com/appeler/ethnicolr/blob/master/ethnicolr/data/census/census_2000.pdf>`__ 
   for what the column names mean.

   ::

      >>> import pandas as pd

      >>> from ethnicolr import census_ln, pred_census_ln

      >>> names = [{'name': 'smith'},
      ...         {'name': 'zhang'},
      ...         {'name': 'jackson'}]

      >>> df = pd.DataFrame(names)

      >>> df
            name
      0    smith
      1    zhang
      2  jackson

      >>> census_ln(df, 'name')
            name pctwhite pctblack pctapi pctaian pct2prace pcthispanic
      0    smith    73.35    22.22   0.40    0.85      1.63        1.56
      1    zhang     0.61     0.09  98.16    0.02      0.96        0.16
      2  jackson    41.93    53.02   0.31    1.04      2.18        1.53


-  **pred\_census\_ln(df, lname_col, year=2000, num\_iter=100, conf\_int=1.0)**

   -  What it does:

      -  Removes extra space.
      -  Uses the `last name census 2000 
         model <ethnicolr/models/ethnicolr_keras_lstm_census2000_ln.ipynb>`__ or 
         `last name census 2010 model <ethnicolr/models/ethnicolr_keras_lstm_census2010_ln.ipynb>`__ 
         to predict the race and ethnicity.


   +--------------+---------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                     |
   +==============+=====================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred        |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string}* name of the column containing the last name                                                |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **year** : *{2000, 2010}, default=2000* year of census to use                                                       |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                           |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=1.0* confidence interval in predicted class                                         |
   +--------------+---------------------------------------------------------------------------------------------------------------------+


   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), api (percentage chance
      asian), black, hispanic, white. For each race it will provide the
      mean, standard error, lower & upper bound of confidence interval

   *(Using the same dataframe from example above)*
   ::

         >>> census_ln(df, 'name')
               name pctwhite pctblack pctapi pctaian pct2prace pcthispanic
         0    smith    73.35    22.22   0.40    0.85      1.63        1.56
         1    zhang     0.61     0.09  98.16    0.02      0.96        0.16
         2  jackson    41.93    53.02   0.31    1.04      2.18        1.53

         >>> census_ln(df, 'name', 2010)
               name   race pctwhite pctblack pctapi pctaian pct2prace pcthispanic
         0    smith  white     70.9    23.11    0.5    0.89      2.19         2.4
         1    zhang    api     0.99     0.16  98.06    0.02      0.62        0.15
         2  jackson  black    39.89    53.04   0.39    1.06      3.12         2.5

         >>> pred_census_ln(df, 'name')
               name   race       api     black  hispanic     white
         0    smith  white  0.002019  0.247235  0.014485  0.736260
         1    zhang    api  0.997807  0.000149  0.000470  0.001574
         2  jackson  black  0.002797  0.528193  0.014605  0.454405


-  **pred\_fl\_reg\_ln(df, lname_col, num\_iter=100, conf\_int=1.0)**

   -  What it does?:

      -  Removes extra space, if there.
      -  Uses the `last name FL registration
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln.ipynb>`__
         to predict the race and ethnicity.

   +--------------+---------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                     |
   +==============+=====================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred        |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **lname_col** : *{string}* name of the column containing the last name                                              |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate the uncertainty                                |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=1.0* confidence interval                                                            |
   +--------------+---------------------------------------------------------------------------------------------------------------------+



   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), asian (percentage chance
      Asian), hispanic, nh\_black, nh\_white. For each race it will provide
      the mean, standard error, lower & upper bound of confidence interval

   ::

      >>> import pandas as pd

      >>> names = [
      ...             {"last": "sawyer", "first": "john", "true_race": "nh_white"},
      ...             {"last": "torres", "first": "raul", "true_race": "hispanic"},
      ...         ]
      
      >>> df = pd.DataFrame(names)

      >>> from ethnicolr import pred_fl_reg_ln, pred_fl_reg_name, pred_fl_reg_ln_five_cat, pred_fl_reg_name_five_cat

      >>> odf = pred_fl_reg_ln(df, 'last', conf_int=0.9)
      ['asian', 'hispanic', 'nh_black', 'nh_white']

      >>> odf
         last first true_race  asian_mean  asian_std  asian_lb  asian_ub  hispanic_mean  hispanic_std  hispanic_lb  hispanic_ub  nh_black_mean  nh_black_std  nh_black_lb  nh_black_ub  nh_white_mean  nh_white_std  nh_white_lb  nh_white_ub      race
      0  Sawyer  john  nh_white    0.009859   0.006819  0.005338  0.019673       0.021488      0.004602     0.014802     0.030148       0.180929      0.052784     0.105756     0.270238       0.787724      0.051082     0.705290     0.860286  nh_white
      1  Torres  raul  hispanic    0.006463   0.001985  0.003915  0.010146       0.878119      0.021998     0.839274     0.909151       0.013118      0.005002     0.007364     0.021633       0.102300      0.017828     0.075911     0.130929  hispanic

      [2 rows x 20 columns]

      >>> odf.iloc[0]
      last               Sawyer
      first                john
      true_race        nh_white
      asian_mean       0.009859
      asian_std        0.006819
      asian_lb         0.005338
      asian_ub         0.019673
      hispanic_mean    0.021488
      hispanic_std     0.004602
      hispanic_lb      0.014802
      hispanic_ub      0.030148
      nh_black_mean    0.180929
      nh_black_std     0.052784
      nh_black_lb      0.105756
      nh_black_ub      0.270238
      nh_white_mean    0.787724
      nh_white_std     0.051082
      nh_white_lb       0.70529
      nh_white_ub      0.860286
      race             nh_white
      Name: 0, dtype: object


-  **pred\_fl\_reg\_name(df, lname_col, num\_iter=100, conf\_int=1.0)**

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name FL
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_name.ipynb>`__
         to predict the race and ethnicity.

   +--------------+-------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                   |
   +==============+===================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred      |
   +--------------+-------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{list}* name of the column containing the name.                                                    |
   +--------------+-------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate the uncertainty                              |
   +--------------+-------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=1.0* confidence interval in predicted class                                       |
   +--------------+-------------------------------------------------------------------------------------------------------------------+


   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), asian (percentage chance
      Asian), hispanic, nh\_black, nh\_white. For each race it will provide
      the mean, standard error, lower & upper bound of confidence interval

   
   *(Using the same dataframe from example above)*
   ::

      >>> odf = pred_fl_reg_name(df, 'last', 'first', conf_int=0.9)
      ['asian', 'hispanic', 'nh_black', 'nh_white']

      >>> odf
         last first true_race  asian_mean  asian_std  asian_lb  asian_ub  hispanic_mean  hispanic_std  hispanic_lb  hispanic_ub  nh_black_mean  nh_black_std  nh_black_lb  nh_black_ub  nh_white_mean  nh_white_std  nh_white_lb  nh_white_ub      race
      0  Sawyer  john  nh_white    0.001534   0.000850  0.000636  0.002691       0.006818      0.002557     0.003684     0.011660       0.028068      0.015095     0.011488     0.055149       0.963581      0.015738     0.935445     0.983224  nh_white
      1  Torres  raul  hispanic    0.005791   0.002906  0.002446  0.011748       0.890561      0.029581     0.841328     0.937706       0.011397      0.004682     0.005829     0.020796       0.092251      0.026675     0.049868     0.139210  hispanic

      >>> odf.iloc[1]
      last               Torres
      first                raul
      true_race        hispanic
      asian_mean       0.005791
      asian_std        0.002906
      asian_lb         0.002446
      asian_ub         0.011748
      hispanic_mean    0.890561
      hispanic_std     0.029581
      hispanic_lb      0.841328
      hispanic_ub      0.937706
      nh_black_mean    0.011397
      nh_black_std     0.004682
      nh_black_lb      0.005829
      nh_black_ub      0.020796
      nh_white_mean    0.092251
      nh_white_std     0.026675
      nh_white_lb      0.049868
      nh_white_ub       0.13921
      race             hispanic
      Name: 1, dtype: object



Application
--------------

To illustrate how the package can be used, we impute the race of the campaign contributors recorded by FEC for the years 2000 and 2010 and tally campaign contributions by race.

- `Contrib 2000/2010 using census_ln <ethnicolr/examples/ethnicolr_app_contrib20xx-census_ln.ipynb>`__
- `Contrib 2000/2010 using pred_census_ln <ethnicolr/examples/ethnicolr_app_contrib20xx.ipynb>`__
- `Contrib 2000/2010 using pred_fl_reg_name <ethnicolr/examples/ethnicolr_app_contrib20xx-fl_reg.ipynb>`__


Data
----------

In particular, we utilize the last-name--race data from the `2000
census <http://www.census.gov/topics/population/genealogy/data/2000_surnames.html>`__
and `2010
census <http://www.census.gov/topics/population/genealogy/data/2010_surnames.html>`__,
the `Wikipedia data <ethnicolr/data/wiki/>`__ collected by Skiena and colleagues,
and the Florida voter registration data from early 2017.

-  `Census <ethnicolr/data/census/>`__
-  `The Wikipedia dataset <ethnicolr/data/wiki/>`__
-  `Florida voter registration database <http://dx.doi.org/10.7910/DVN/UBIG3F>`__

Authors
----------

Rajashekar Chintalapati, Suriyan Laohaprapanon, and Gaurav Sood

Contributor Code of Conduct
---------------------------------

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the `Contributor Code of
Conduct <http://contributor-covenant.org/version/1/0/0/>`__.

License
----------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.
