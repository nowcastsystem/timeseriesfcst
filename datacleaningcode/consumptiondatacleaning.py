import numpy as np
import pandas as pd
rawdatapath = "/Users/you/Desktop/quant/stockdataanalysis/raw/"
predictorpath = rawdatapath + 'predictor.csv'


rawdata = pd.read_csv(rawdatapath+'data.csv')

outcome = rawdata.iloc[:, [0, 2]].to_csv(rawdatapath + 'endprice.csv', index=False)

predictor = rawdata.iloc[:, [1,3,4,5]].to_csv(rawdatapath + 'predictors.csv', index=False)



rawdatapath = "/Users/you/Desktop/energyconsumption/raw/"
predictorpath = rawdatapath + 'predictor.csv'


rawdata =pd.read_csv(rawdatapath+'household_power_consumption.txt', sep=";")

rawdata['dtindex'] = rawdata.Date + ' ' + rawdata.Time


outcome = rawdata.iloc[:, [-1, 4]][2070000:].to_csv(rawdatapath + 'outcome.csv', index=False)

predictor = rawdata.iloc[:, [-1,2,3,5]][2070000:].to_csv(rawdatapath + 'predictors.csv', index=False)



