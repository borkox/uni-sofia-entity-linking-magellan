# Entity Linking with Magellan
This is a project for Sofia University.

Authors:
 - Kiril Dimov
 - Borislav Markov
## Intro
...
 
## Dataset
For dataset we chose a one from AnHai's group about electronics:
https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository
ID:8

We have downloaded the CSV files in this repository in folder `dataset/`
## Installation
Install is done, following steps from here:
http://anhaidgroup.github.io/py_entitymatching/v0.3.x/user_manual/installation.html

`pip install -U numpy scipy py_entitymatching`
### Requirements
- Python 3.x
- Jupyter notebook
- Magellan Entity Matcher
- Other(inferred from Entity Matcher)
  * C Compiler Required (This is necessary because this package contains Cython files)
  * pandas (provides data structures to store and manage tables)
  * scikit-learn (provides implementations for common machine learning algorithms)
  * joblib (provides multiprocessing capabilities)
  * pyqt5 (provides tools to build GUIs)
  * py_stringsimjoin (provides implementations for string similarity joins)
  * py_stringmatching (provides a set of string tokenizers and string similarity functions)
  * cloudpickle (provides functions to serialize Python constructs)
  * pyprind (library to display progress indicators)
  * pyparsing (library to parse strings)
  * six (provides functions to write compatible code across Python 2 and 3)
  * xgboost (provides an implementation for xgboost classifier)
  * pandas-profiling (provides implementation for profiling pandas dataframe)
  * pandas-table (provides data exploration tool for pandas dataframe)
  * openrefine (provides data exploration tool for tables)
  * ipython (provides better tools for displaying tables in notebooks)
  * scipy (dependency for skikit-learn)
