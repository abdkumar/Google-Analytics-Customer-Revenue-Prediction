# Google Analytics Customer Revenue Prediction 

Predict how much GStore customers will spend
.

## Overview
The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.


The challenge is to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer. Hopefully, the outcome will be more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.
## Data
The dataset is available [here](https://www.kaggle.com/c/ga-customer-revenue-prediction/data)
## Installation
If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included.
This project requires the following Python libraries installed:
* sci-kit learn
```bash
conda install scikit-learn
```
* lightgbm
```bash
conda install lightgbm
```
* xgboost
```bash
conda install xgboost
```
* catboost
```bash
conda install catboost
```
* Bayes Optimization
```bash
pip install bayesian-optimization
```
## Code
- ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `Note: since iterative process of preproceeing taken, the cell numbers might not be in order, but the code cells are in order`
```diff
+ Full code is available in customer_revenue_prediction.ipynb notebook.
```
Download the data from the above given link.
Install Anaconda and necessary packages.

The following steps are taken to analyse and apply Machine Learning model
### Preprocessing
 * Some columns namely `device`,`geoNetworkDomain`,`traficSource`,`totals` in the dataset are in JSON form. We need to convert this JSON columns to tabular format. This is done using a`json_normalize` module.
* Created subcolumns for each JSON columns.
* check number of unique values in each column and `drop` constant columns.
This is done using
```python
pd.nunique(dropna=False) 
```
* Explore the target variable i.e, `totals.transactionRevenue`
### Feature Engineering
* Created new columns (`_day`,`_weekday`,`_month`,`_year`) from `date` feature.
* created `_visitHour` feature from the `visitStartTime` which is given in timestamp format.
### Exploratory Data Analysis

_Date_ and _Time_ v/s_transactionRevenue_
* visalised the change of `transactionRevenue` against `_day`,`_weekday`,`_month`,`_year`.
* visualised the change of `transactionRevenue` against `_visitHour`.

_device_ columns v/s_transactionRevenue_
* visualised the change of `transactionRevenue` against `device.browser`, `device.deviceCategory`, `device.operatingSystem`. 

_geoNetwork_ v/s _transactionRevenue_
* visualised the change of `transactionRevenue` against `geoNetwork.continent`, `geoNetwork.country`, `geoNetwork.subContinent` , `geoNetwork.networkDomain`.

_trafficSource_ v/s _transactionRevenue_
* visualised  the change of`transactionRevenue` against `trafficSource.source`,`trafficSource.referralPath', `trafficSource.medium`.

_totals_ v/s _transactionRevenue_
* visualised  the change of`transactionRevenue` against `totals.pageViews`, `totals.hits`.
## Missing value treatment
```python
pd.fillna(value)
```
* Numerical columns
In numerical columns list only `totals.hits`,`totals.bounces` and `totals.pageViews` were missing, so filled it with appropriate value.

* Categorical columns
some columns had more than 60% of missing value so not able to fill it with existing category. Hence filled it with a new category *'unknown'*.

## Label Encoding
All categorical columns  are encoded using `LabelEncoder()` class which is imported from `sklearn.preprocessing` module.

## Train and Validation split
* train_v2.csv dataset has covered the data from 1st August 2016 to 30th April 2018.
* Last 4 months data i.e, 1st Jan 2018 to 30th April 2018 as validation set.

## Training

* I have used the `LightGBM` model.
* It supports parallel and `GPU` learning. 
* It is highly efficeint in handling large size data.
* To enable GPU access we need to install some drivers
* for more information check [here](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)

## Hyper parameter tuning
_This is the most important step in the modelling to boost the model performace_.
 _It is a time consuming process_
`Bayesian Optimization` is used for tuning.
* well known tuning methods like `GridSeachCV` and `RandomizedSearchCV` doesn't keep track of previous error, so not a better choice for perfect tuning.
* `Bayesian Optimization` tends to find good points in search space with relatively few function evaluations.
* selected some important parameters of `LightGBM` for tuning, because to tune all parameters it takes lot of time.

## Cross Validation
* Cross Validation is used to assess the predictive performance of the models and and to judge how they perform outside the sample to a new data set also known as test data
* Used `n_folds = 5`

## Evaluation of a Validation set
* find the `root mean square error`(rmse) of actual transactionRevenue and predicted transactionRevenue. 
* check whether model is optimised or not.

## Ensembling
* I tried to ensemble `LightGBM`, `xgboost`, `catboost` but performance did not improve much(lack of time to tune paremeters of xgboost and catboost).

## Submission
* The predicted transactionRevenue of test set will be in `np.log1p` form, so take inverse i,e apply `np.expm1`.
* calculate total _transactionRevenue_ of each visitor.
```python
import pandas as pd
sub_df = sub_df.groupby('fullVisitorID')['transactionRevenue'].sum().reset_index()
```
* apply `np.log1p` to the _transactionRevenue_ Series.
* change the column name of _transactionRevenue_ to _PredictedLogRevenue_
* save the dataframe to csv format using 
```python
sub_df.to_csv(path, index=False)
```
## Public Leaderboard score
_My best score on kaggle for this competition is_
```diff 
root mean square error 1.7061
```
