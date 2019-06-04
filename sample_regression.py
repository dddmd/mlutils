import mlutils.artgor_utils
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


# bostonデータ読み込み
from sklearn import datasets
boston = datasets.load_boston()
X_train = pd.DataFrame(boston.data, columns=boston.feature_names) # 説明変数(data)
y_train = pd.Series(boston.target) # 目的変数(target)追加

# CrossValidation
n_fold = 3
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

# XGBoost
xgb_params = {'eta': 0.03,
              'max_depth': 9,
              'subsample': 0.85,
              'colsample_bytree': 0.3,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
              'nthread': -1}


result_dict_xgb = mlutils.artgor_utils.train_model_regression(X=X_train, X_test=X_train, y=y_train, params=xgb_params, folds=folds, model_type='xgb', plot_feature_importance=True)


