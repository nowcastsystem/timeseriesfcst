# This file contains functions to combine data from different sources together.

import pandas as pd
import QUANTAXIS as qa
import pymongo
import json
import datetime

def readdata(code,start, end):
    '''This function loads data from local mongodb using qafetch function from quantaxis.
        Args:
            code: stock code, string type.
            start: start date.
            end: end date.
        Returns:
            A data frame indexed with standard datetime index.
    '''
    data = qa.QA_fetch_stock_day_adv(code=code, start=start, end=end)
    result = data.data
    result = result.sort_index(ascending=False)
    result = result.reset_index(level=1)
    result = result.drop(columns='code')

    return result

def fillinmissing(data, dtindex, fillin=None, indicator=True):
    '''This function takes a data frame that is indexed by standard datetime index.
    It completes the data frame by encoding values to missing records.
        Args:
            data: a data frame that is indexed by datetime index with missing records to be filled in.
            dtindex: a full datetime index list as a reference to locate the missing records.
            fillin: indicate what value should be filled in.
            indicator: if is True. The function will add an additional column indicts which row is newly filled in.
        Returns:
            A data frame without missing records.
    '''
    fulldata = pd.DataFrame(index=dtindex)
    fulldata = fulldata.join(data)

    if indicator is True:
        ismissing = pd.notna(fulldata)
        fulldata = fulldata.fillna(fillin)
        return fulldata, ismissing

    return fulldata


def get_dtindex(start,end):
    dtindex = pd.date_range(start=start,
                            end=end,
                            freq='D')
    return dtindex


def get_parameters(code,start, end):
    outcome = readdata(code=code, start=start, end=end)
    dtindex = get_dtindex(start=start, end=end)
    outcome= fillinmissing(data=outcome,
                                       dtindex=dtindex,
                                       fillin=0,
                                       indicator=True)

    return outcome, dtindex

def date2str(data):

    if 'date' in data.columns:
        data.date = data.date.apply(str)
    return json.loads(data.to_json(orient='records'))

def upload_data(code,start,end):
    outcome, dtindex= get_parameters(code=code, start=start, end=end)
    #connect to mongodb
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    database = myclient['mydatabase']
    datacol = database[code+str(datetime.date.today())]
    # dtindexcol = database[code+str(datetime.date.today())+'index']
    outcome = outcome[0]
    outcome['date'] = outcome.index
    outcome = date2str(outcome)
    datacol.insert(outcome)
    # dtindexcol.insert(json.loads(outcome.T.to_json()).values())