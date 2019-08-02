import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.metrics import mean_squared_error


import argparse  # This package is used to read in arguments from command line.
import configparser  # This package is used to read in configuration files
import pandas as pd
import xgboost as xgb
import os
import pickle

from timeseriesfcst import dataprep
from timeseriesfcst import featureengineering as fe
from timeseriesfcst import xgbinput
# import xgbinput

from catboost import CatBoostRegressor, Pool, cv
from catboost import CatBoost
from sklearn.metrics import accuracy_score

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')


outcome, isencoded, predictors, dtindex = dataprep.get_parameters(config=config)
# print(predictors)

fulldf = fe.get_fulldf(outcome=outcome, predictors=predictors, dtindex=dtindex)

fulldf = fulldf.dropna()

xgbinpt = xgbinput.XGBInput(label=fulldf['Traffic'],
                            covariate=fulldf.drop(columns='Traffic'),
                            splitdt='2018-09-01 00:00:00')



def xgb_rmse(max_depth=3, eta=0.01):
    hyperparameters = {
        'max_depth': max_depth,
        'eta': eta,
        'silent': 1,
        'objective': 'reg:squarederror',
        'nthread': 5,
        'eval_metric': 'rmse',
        'numround': 5000
    }

    def objective(hyperparameters):
        model = xgb.XGBRegressor(**hyperparameters)

        model.fit(xgbinpt.covariatetrain, xgbinpt.labeltrain, eval_metric='rmse')

        prediction = model.predict(xgbinpt.covariatetest)

        rmse = np.sqrt(np.mean(np.square(prediction - xgbinpt.labeltest)))
        return -rmse

    return objective(hyperparameters)


def optimize_xgb():
    """Apply Bayesian Optimization to Random Forest parameters."""
    def xgb_wrapper(max_depth, eta):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return xgb_rmse(int(max_depth), eta)

    optimizer = BayesianOptimization(
        f=xgb_wrapper,
        pbounds={
            "max_depth": (2, 5),
            "eta": (0.01, 0.5)
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=40)

    print("Final result:", optimizer.max)

print(Colours.green("--- Optimizing XGBoost ---"))
optimize_xgb()