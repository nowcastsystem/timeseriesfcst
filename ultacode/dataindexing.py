# This file is to index the raw data.
# Each data need to be indexed by a column 'dtindex' with standard datetime values.

import pandas as pd
import numpy as np

rawdatapath = "/Users/you/Desktop/ultaanalysis/raw/"

Trafficpath = rawdatapath + "TrafficWeatherMatrix" + ".csv"
ultatraffic = pd.read_csv(Trafficpath)
columnnamestraffic = ['Traffic', 'Store']

def ultatrafficindexing(ultatraffic):
    dtindex = pd.to_datetime(ultatraffic['Day'], format="%Y%m%d")
    dtindex = dtindex + pd.to_timedelta(ultatraffic['Hour'] - 1, unit='h')
    return dtindex

ultatraffic['dtindex'] = ultatrafficindexing(ultatraffic)
ultatraffic[['dtindex'] + columnnamestraffic].to_csv(rawdatapath + "traffic.csv", index=False)


Weatherpath = rawdatapath + "weatherIndicatorPrevHour" + ".csv"
ultaweather = pd.read_csv(Weatherpath)
columnnamesweather = ['Temp', 'Pressure', 'Humidity', 'Windspeed',
                      'Winddeg', 'CloudCover', 'Rain', 'Snow', 'Store']

def ultaweatherindexing(ultaweather):
    dtindex = pd.to_datetime(ultaweather['Day'], format="%Y%m%d")
    dtindex = dtindex + pd.to_timedelta(ultaweather['Hour'] - 1, unit='h')
    return dtindex

ultaweather['dtindex'] = ultaweatherindexing(ultaweather)
ultaweather[['dtindex'] + columnnamesweather].to_csv(rawdatapath + "weather.csv", index=False)


holidaypath = rawdatapath + "holidays.csv"
holiday = pd.read_csv(holidaypath)

def holidayindexing(holiday):
    dtindex = pd.to_datetime(holiday.Day, format="%Y%m%d")
    fulldt = pd.date_range(start='2017-1-1', end='2019-3-1', freq='H')

    holiday= pd.DataFrame(index=fulldt)
    holiday['isholiday'] = np.isin(fulldt, dtindex)
    holiday['dtindex'] = fulldt
    return holiday

holiday = holidayindexing(holiday)
holiday.to_csv(rawdatapath + "holiday.csv", index=False)
