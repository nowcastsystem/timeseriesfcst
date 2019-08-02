# This code splits data into subsets.

import pandas as pd
import numpy as np

rawdatapath = "/Users/you/Desktop/ultaanalysis/raw/"
trafficpath = rawdatapath + 'traffic.csv'
weatherpath = rawdatapath + 'weather.csv'

rawdata = pd.read_csv(trafficpath)
rawdata.groupby('Store')
for id, value in rawdata.groupby('Store'):
    value.drop(columns='Store').to_csv(rawdatapath+'traffic'+str(id)+".csv", index=False)


rawdata = pd.read_csv(weatherpath)
rawdata.groupby('Store')
for id, value in rawdata.groupby('Store'):
    value.drop(columns='Store').to_csv(rawdatapath+'weather'+str(id)+".csv", index=False)
