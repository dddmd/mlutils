import optuna
from mlutils import artgor_utils
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

def objective(trial):
    #params = {'num_leaves': 128,
    #      'min_child_samples': 79,
    #      'objective': 'gamma',
    #      'max_depth': -1,
    #      'learning_rate': 0.01,
    #      "boosting_type": "gbdt",
    #      "subsample_freq": 5,
    #      "subsample": 0.9,
    #      "bagging_seed": 11,
    #      "metric": 'mae',
    #      "verbosity": -1,
    #      'reg_alpha': 0.1302650970728192,
    #      'reg_lambda': 0.3603427518866501,
    #      'colsample_bytree': 0.2
    #     }
    search_params = {
        'num_leaves': trial.suggest_int('num_leaves', 23, 128),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 17),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 90),
        'objective': 'gamma',
        #'objective':'regression',
        'max_depth': trial.suggest_int('max_depth', -1, 21),
        "learning_rate": trial.suggest_uniform('learning_rate', 0.001, 0.01),
        "boosting": "gbdt",
        "subsample_freq": trial.suggest_int('min_data_in_leaf', 0, 8),
        "subsample": trial.suggest_uniform("subsample", 0.8, 1.0),
        "bagging_freq": trial.suggest_int('min_data_in_leaf', 0, 8),
        "bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.8, 1.0),
        "bagging_seed": 11,
        "metric": 'mae',
        "reg_alpha": trial.suggest_uniform('reg_alpha', 0.0, 0.2),
        "reg_lambda": trial.suggest_uniform('reg_lambda', 0.0, 0.4),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.2, 1.0),
        "min_child_weight": trial.suggest_int('min_child_weight', 1, 100),
        "verbosity": -1,
        #"nthread": -1,
        "random_state": 42}   
    
    result_dict_lgb_opt = artgor_utils.train_model_regression(X=X_train, X_test=X_train, y=y_train, params=search_params, folds=folds, model_type='lgb',
                                                                                  eval_metric='mae', plot_feature_importance=False)
    return np.mean(result_dict_lgb_opt['scores'])

study = optuna.create_study()
study.optimize(objective, n_trials=10)