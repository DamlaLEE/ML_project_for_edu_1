import os
import pickle
import numpy as np
import pandas as pd
import missingno #결측치
import seaborn as sns
import matplotlib.pyplot as plt

# ML model
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#예측 검증 지표
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import KFold

#cross_validation
from sklearn.model_selection import GridSearchCV

# **step1** split data (for train & test)
def split_data(df):
    return df[:800], df[800:]

# step3 feature importance
def feature_importance_check(model, X_train, y_train):
    fitting_model = model(n_estimators=100, random_state=42)
    fitting_model.fit(X_train, y_train)

    feature_importance = fitting_model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
                'feature' : feature_names,
                'Importance' : feature_importance.round(2),
                            }).set_index('feature').sort_values(by="Importance", ascending=False)
    
    return importance_df

#step4 Cross validation
def rmse_scorer(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))

