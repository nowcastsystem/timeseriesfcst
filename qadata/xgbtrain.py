import xgboost as xgb
from pandas.plotting import register_matplotlib_converters
import pymongo
import datetime
import pandas as pd


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
    fullcol = database[code+'fulldf'+str(datetime.date.today())]
    cursor = fullcol.find()
    fulldf = pd.DataFrame(list(cursor))
    fulldf = fulldf.drop(columns='_id')
    fulldf.set_index('date', inplace=True)
    return fulldf

def model_train(code):
    fulldf = get_fulldf(code=code)
    xgbinpt = XGBInput(label=fulldf['close'], covariate=fulldf.drop(columns='close'), splitdt='2019-03-01 00:00:00')
    param = {
        'max_depth': 3,
        'eta': .01,
        'silent': 1,
        'objective': 'reg:squarederror',
        'nthread': 5,
        'eval_metric': 'rmse'
    }

    numround = 5000
    esr = 10
    register_matplotlib_converters()

    xgbmod = xgb.train(param, xgbinpt.dtrain, numround, xgbinpt.evallist, early_stopping_rounds=esr)

    filename =  code + str(datetime.date.today()) + '.model'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(xgbmod, f)