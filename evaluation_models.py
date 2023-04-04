# Create a file named evaluate.py that contains the following functions.
#     - plot_residuals(y, yhat): creates a residual plot
#     - regression_errors(y, yhat): returns the following values:
#         - sum of squared errors (SSE)
#         - explained sum of squares (ESS)
#         - total sum of squares (TSS)
#         - mean squared error (MSE)
#         - root mean squared error (RMSE)
#     - baseline_mean_errors(y): computes the SSE, MSE, and RMSE for the baseline model
#     - better_than_baseline(y, yhat): returns true if your model performs better than the baseline, otherwise false
# 

# Imports
# data
import wrangle
#standard ds imports
import pandas as pd
import numpy as np
#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
# stats
from statsmodels.formula.api import ols
from sklearn.metrics import r2_score

# Acquire data
train, validate, test = wrangle.wrangle_zillow()

# Prep data
# define independent and X, and dependent and y variables
X = independent_var = train.sqr_ft
X_var = 'sqr_ft'
y = dependent_var = train.tax_value
y_var = 'tax_value'

# create baseline column
def create_baseline():
    y = dependent_var
    train['baseline'] = y.mean()
    return train

# fit ols model and create yhat
def fit_ols_and_create_yhat(y, X):
    y = y_var
    X = X_var
    model = ols(f'{y} ~ {X}', data=train).fit()
    train['yhat'] = model.predict(independent_var)
    return train

# create definition that returns the residual
def residual():
    r = y - train.yhat
    train['residual'] = r
    return train

# create a scatter plot of residuals
def plot_residuals(x_label, y_label):
    sns.scatterplot(data= train, x= X_var, y= 'residual')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt.show()

# calculate and return regression errors
def regression_errors():
    train['yhat_residual'] = y - train.yhat
    train['residual_squared'] = train.yhat_residual**2
    SSE = sum(train.residual_squared)
    MSE = SSE/len(train)
    RMSE = MSE**.5
    ESS = sum((train.yhat - y.mean())**2)
    TSS = sum((y - y.mean())**2)
    return SSE, MSE, RMSE, ESS, TSS

# calulate and return baseline errors
def baseline_errors():
    train['baseline_residual'] = y - train.baseline
    train['baseline_residual_squared'] = train.baseline_residual**2
    SSE_baseline = sum(train.baseline_residual_squared)
    MSE_baseline = SSE_baseline/len(train)
    RMSE_baseline = MSE_baseline**.5
    return SSE_baseline, MSE_baseline, RMSE_baseline

# is the model better than baseline
def better_than_baseline():
    r2 = r2_score(y, train.yhat)
    if SSE < SSE_baseline:
        print('The model performs better than baseline')
    else:
        print('The model does not perform better than the baseline')
    return print(f'R2: {r2}')