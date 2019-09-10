import argparse #This package is used to read in arguments from command line.
import configparser #This package is used to read in configuration files
import pandas as pd
import xgboost as xgb
import os
import pickle
import plotly.graph_objs as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.metrics import r2_score

import base64
import datetime
import io


import dash_table
import pandas as pd

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


outcomelag = fe.get_lag(data=outcome, lags=range(3, 30), unit='D')
outcomelagmean = fe.get_lag_mean(data=outcome, lags=range(3, 30), unit='D', meanby='D')

#predictorslag = fe.get_lag(data=predictors, lags=range(1, 10), unit='D')
#predictorslagmean = fe.get_lag_mean(data=predictors, lags=range(1, 10), unit='D', meanby='D')

#predictorslaglagbyh = fe.get_lag(data=predictors, lags=range(1, 5), unit='H')

datetimefeature = fe.gettimefeature(dtindex)

#featurelist = [outcomelag, outcomelagmean, predictorslag, predictorslagmean, datetimefeature]
#fullfeature = pd.concat(featurelist, axis=1)

fulllist = [outcome, outcomelag, outcomelagmean, datetimefeature]
fulldf = pd.concat(fulllist, axis=1).dropna()
fulldf = fulldf[fulldf['close'] != 0]

splitdt = config['traintestsplit']['splitdt']

xgbinpt = xgbinput.XGBInput(label=fulldf['close'], covariate=fulldf.drop(columns='close'), splitdt=splitdt)

outcome = fulldf.iloc[:, 0].copy()
covariate = fulldf.iloc[:, 1:].copy()



outcometrain = outcome.loc[outcome.index <= splitdt].copy()
outcometest = outcome.loc[outcome.index > splitdt].copy()

covariatetrain = covariate.loc[covariate.index <= splitdt].copy()
covariatetest = covariate.loc[covariate.index <= splitdt].copy()



modelinputpath = config['modelinputpath']['path'] + 'stock.xgbdata'
os.makedirs(os.path.dirname(modelinputpath), exist_ok=True)

hyperparameters = {
    'max_depth': 5,
    'eta': .03,
    'silent': 1,
    'objective': 'reg:squarederror',
    'nthread': 5,
    'eval_metric': 'rmse',
    'num_boost_round': 1000,
    'early_stopping_rounds': 50
}

model = xgb.XGBRegressor(**hyperparameters)

model.fit(covariatetrain, outcometrain, eval_metric='rmse')
prediction = model.predict(covariate)

filename = config['modelpath']['path']+'stockexample' + '.model'
os.makedirs(os.path.dirname(filename), exist_ok=True)

with open(filename, "wb") as f:
    pickle.dump(model, f)




r2 = r2_score(y_true=outcome, y_pred=prediction)



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([
    dcc.Input(id='my-id', value='initial value', type='text'),
    dcc.Graph(id='timeseriesplot'),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),

])


@app.callback(
    Output(component_id='timeseriesplot', component_property='figure'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):

    filename = config['modelpath']['path']+'stockexample' + '.model'
    xgbmod = pickle.load(open(filename, "rb"))

    r2 = r2_score(y_true=outcome, y_pred=xgbmod.predict(covariate))

    prediction_val = go.Scatter(
        x=outcome.index,
        y=xgbmod.predict(covariate),
        name="Predicted Value",
        line=dict(color='#17BECF'),
        opacity=0.8)

    actual_val = go.Scatter(
        x=outcome.index,
        y=outcome,
        name="Actual Value",
        line=dict(color='#7F7F7F'),
        opacity=0.8)

    data = [prediction_val, actual_val]

    layout = dict(
        title='Time Series Forcast: A Simple Demo' + ", r2=" + str(round(r2, 2)),

        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type='category'
        ),
        shapes=[
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'paper',
                'x0': np.where(xgbinpt.traintestindex > pd.to_datetime(splitdt))[0][0],
                'y0': 0.08,

                'x1': np.where(xgbinpt.traintestindex > pd.to_datetime(splitdt))[0][0],
                'y1': 0.9,
                'line': {
                    'color': 'rgb(55, 128, 191)',
                    'width': 3}
            }
        ],

    )

    fig = dict(data=data, layout=layout)

    return fig



def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



if __name__ == '__main__':
    app.run_server(debug=True)
