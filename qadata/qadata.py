import argparse  # This package is used to read in arguments from command line.
import configparser  # This package is used to read in configuration files
import pandas as pd
import xgboost as xgb
import os
import pickle


import dataprep
import featureengineering as fe
import xgbtrain
import catbtrain


code = '000002'
start = '2009-07-13'
end = '2019-07-12'

dtindex = pd.date_range(start=start,end=end,freq='D')

dataprep.upload_data(code=code,start=start,end=end)
print('uploaddata')
fe.upload_feature(code=code,dtindex=dtindex)

#xgboost
xgbtrain.model_train(code)

#catboost
# catbtrain.model_train(code)