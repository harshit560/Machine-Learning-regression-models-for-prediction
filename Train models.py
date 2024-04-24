#!/usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.svm import SVR
from sklearn import linear_model
from cubist import Cubist
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r'C:\\Users\\Hp\\Desktop\\sample_data.csv')

# Convert DataFrame to JSON format
json_data = data.to_json(orient='records')

# Load JSON data into a DataFrame
df = pd.read_json(json_data, orient='records')

#Used later for destandarize
y1 = df['prediction_param']
df1 = pd.DataFrame(y1, columns=['prediction_param'])

# Data Scaling
df_norm = (df - df.min()) / (df.max() - df.min())

# Split data into train and test sets
y = df_norm['prediction_param']
X = df_norm.drop('prediction_param', axis=1)
#keep training data % on the higher side as models will be saved and later tested using the values outside of the databse.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

def train_models(X_train, y_train):
    # Random Forest Model
    model_rf = RandomForestRegressor()
    regr_trans_rf = TransformedTargetRegressor(regressor=model_rf, transformer=QuantileTransformer(output_distribution='normal'))
    regr_trans_rf.fit(X_train, y_train)

    # XGBoost Model
    model_xg = xg.XGBRFRegressor()
    regr_trans_xg = TransformedTargetRegressor(regressor=model_xg, transformer=QuantileTransformer(output_distribution='normal'))
    regr_trans_xg.fit(X_train, y_train)

    # Support Vector Regressor Model
    param_grid_svr = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
    grid_svr = GridSearchCV(SVR(), param_grid_svr, refit=True, verbose=4, n_jobs=-1)
    regr_trans_svr = TransformedTargetRegressor(regressor=grid_svr, transformer=QuantileTransformer(output_distribution='normal'))
    regr_trans_svr.fit(X_train, y_train)

    # Linear Regression Model
    reg_model_lr = linear_model.LinearRegression()
    reg_model_lr.fit(X_train, y_train)

    # Cubist Model
    param_grid_cubist = {'n_committees': [100, 200, 300], 'random_state': [80], 'verbose': [8]}
    grid_cubist = GridSearchCV(Cubist(), param_grid_cubist, cv=12, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_cubist.fit(X_train, y_train)
    best_model_cubist = grid_cubist.best_estimator_

    return regr_trans_rf, regr_trans_xg, regr_trans_svr, reg_model_lr, best_model_cubist

# Training models
trained_models = train_models(X_train, y_train)
