# This file contains functions to combine data from different sources together.

import pandas as pd

def readdata(path):
    '''This function reads in a data frame with index 'dtindex' from a specified path.
        Args:
            path: a path that specifies the location of the data.
        Returns:
            A data frame indexed with standard datetime index. The column name of that index must be dtindex.
    '''

    data = pd.read_csv(path)
    data.index = data['dtindex']

    return data.drop(columns='dtindex')

def fillinmissing(data, dtindex, fillin=None, indicator=False):
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

