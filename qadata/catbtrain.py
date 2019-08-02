#catboost

from catboost import CatBoostRegressor, Pool, cv
from catboost import CatBoost
from sklearn.metrics import accuracy_score

import xgboost as xgb

from pandas.plotting import register_matplotlib_converters
import pymongo
import datetime
import pandas as pd
import os
import pickle


class XGBInput(object):
    def __init__(self, label, covariate, splitdt):
        self.label = label
        self.covariate = covariate

        self.labeltrain = self.label.loc[self.label.index <= splitdt].copy()
        self.covariatetrain = self.covariate.loc[self.covariate.index <= splitdt].copy()

        self.labeltest = self.label.loc[self.label.index > splitdt].copy()
        self.covariatetest = self.covariate.loc[self.covariate.index > splitdt].copy()

        self.dtrain = xgb.DMatrix(self.covariatetrain, label=self.labeltrain)
        self.trainindex = self.covariatetrain.index
        self.dtest = xgb.DMatrix(self.covariatetest, label=self.labeltest)
        self.testindex = self.covariatetest.index
        self.evallist = [(self.dtest, 'eval'), (self.dtrain, 'train')]

        self.dtraintest = xgb.DMatrix(covariate, label=label)
        self.traintestindex = covariate.index

    def gettrain(self):
        return self.dtrain, self.covariatetrain, self.labeltrain

    def gettest(self):
        return self.dtest, self.covariatetest, self.labeltest


def get_fulldf(code):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    database = myclient['mydatabase']
    fullcol = database[code + 'fulldf' + str(datetime.date.today())]
    cursor = fullcol.find()
    fulldf = pd.DataFrame(list(cursor))
    fulldf = fulldf.drop(columns='_id')
    fulldf['date'] = pd.to_datetime(fulldf['date'])
    fulldf.set_index('date', inplace=True)
    return fulldf


def model_train(code):
    fulldf = get_fulldf(code=code)
    xgbinpt = XGBInput(label=fulldf['close'], covariate=fulldf.drop(columns='close'), splitdt='2019-03-01 00:00:00')

    numround = 5000
    esr = 10
    register_matplotlib_converters()

    # xgbmod = xgb.train(param, xgbinpt.dtrain, numround, xgbinpt.evallist, early_stopping_rounds=esr)

    # filename = './/' + code + str(datetime.date.today()) + '.model'
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    #
    # with open(filename, "wb") as f:
    #     pick.dump(xgbmod, f)

    param = {
        'depth': 2,
        'learning_rate': .05,
        'verbose': 'False',
        'loss_function': 'RMSE',
        'thread_count': 5,
        'eval_metric': 'RMSE',
        'boosting_type':'Plain'
    }
    model = CatBoostRegressor(depth=3,learning_rate=0.01,loss_function='RMSE',thread_count=5,eval_metric='RMSE',boosting_type='Ordered',
        random_seed=42,iterations=5000
    )

    # print(list(predictors.columns))

    model.fit(
        xgbinpt.covariatetrain, xgbinpt.labeltrain,
        # cat_features=list(predictors.columns),
        eval_set=(xgbinpt.covariatetest, xgbinpt.labeltest),
    #     logging_level='Verbose',  # you can uncomment this for text output
    #     plot=True
    );
    filename = './/catb' + code + str(datetime.date.today()) + '.model'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        pick.dump(model, f)

