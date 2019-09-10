import argparse #This package is used to read in arguments from command line.
import configparser #This package is used to read in configuration files
import pandas as pd
import xgboost as xgb
import os
import pickle

from timeseriesfcst import dataprep
from timeseriesfcst import featureengineering as fe
from timeseriesfcst import xgbinput
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')

'''
get arguments from cli, run:
python3 setup.py install
python3  timeseriesfcst -c /Users/you/Desktop/timeseriesfcst_config/config.ini

# create a new argument parser
parser = argparse.ArgumentParser(description="Simple argument parser")
# add a new command line option, call it '-c' and set its destination to 'config'
parser.add_argument("-c", action="store", dest="config_file")
# get the result
result = parser.parse_args()

# read configuration from configuration file.
config = configparser.ConfigParser()
config.read(result.config_file)
'''


dtindex = pd.date_range(start=config['dtindex']['startdt'],
                        end=config['dtindex']['enddt'],
                        freq=config['dtindex']['by'])


outcome = dataprep.readdata(config['outcomepath']['path'])['close']
outcome, isencoded = dataprep.fillinmissing(data=outcome,
                                 dtindex=dtindex,
                                 fillin=0,
                                 indicator=True)

predictors = pd.DataFrame(index=dtindex)


for i in config['predictorspath']:
    predictor = dataprep.readdata(config['predictorspath'][i])

    predictor = dataprep.fillinmissing(data=predictor,
                                       dtindex=dtindex,
                                       fillin=None)
    predictors = predictors.join(predictor)


outcomelag = fe.get_lag(data=outcome, lags=range(3, 10), unit='D')
outcomelagmean = fe.get_lag_mean(data=outcome, lags=range(3, 10), unit='D', meanby='D')

#predictorslag = fe.get_lag(data=predictors, lags=range(1, 10), unit='D')
#predictorslagmean = fe.get_lag_mean(data=predictors, lags=range(1, 10), unit='D', meanby='D')

#predictorslaglagbyh = fe.get_lag(data=predictors, lags=range(1, 5), unit='H')

datetimefeature = fe.gettimefeature(dtindex)

#featurelist = [outcomelag, outcomelagmean, predictorslag, predictorslagmean, datetimefeature]
#fullfeature = pd.concat(featurelist, axis=1)

fulllist = [outcome, outcomelag, outcomelagmean, datetimefeature]
fulldf = pd.concat(fulllist, axis=1).dropna()
fulldf = fulldf[fulldf['close'] != 0]



xgbinpt = xgbinput.XGBInput(label=fulldf['close'], covariate=fulldf.drop(columns='close'), splitdt='2019-01-01 00:00:00')


modelinputpath = config['modelinputpath']['path'] + 'stock.xgbdata'
os.makedirs(os.path.dirname(modelinputpath), exist_ok=True)



param = {
    'max_depth': 4,
    'eta': .005,
    'silent': 1,
    'objective': 'reg:squarederror',
    'nthread': 5,
    'eval_metric': 'rmse'
}


numround = 10000
#early_stopping_rounds = 100
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()

xgbmod = xgb.train(param, xgbinpt.dtrain, numround, xgbinpt.evallist,early_stopping_rounds=10)

filename = config['modelpath']['path']+'stockexample' + '.model'
os.makedirs(os.path.dirname(filename), exist_ok=True)

with open(filename, "wb") as f:
    pickle.dump(xgbmod, f)

