import base64
import datetime
import io
import configparser
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
import numpy as np

import dataprep
import featureengineering as fe











config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('config.ini')






external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
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
    dcc.Graph(id='timeseriesplot'),
])


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


def parse_contents_getdf(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df.index = df['dtindex']
            return df.drop(columns='dtindex')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])





def timeseriesfs(df):
    dtindex = pd.date_range(start=config['dtindex']['startdt'],
                            end=config['dtindex']['enddt'],
                            freq=config['dtindex']['by'])
    outcome = df
    outcome, isencoded = dataprep.fillinmissing(data=outcome,
                                                dtindex=dtindex,
                                                fillin=0,
                                                indicator=True)

    outcomelag = fe.get_lag(data=outcome, lags=range(3, 30), unit='D')
    outcomelagmean = fe.get_lag_mean(data=outcome, lags=range(3, 30), unit='D', meanby='D')
    datetimefeature = fe.gettimefeature(dtindex)

    fulllist = [outcome, outcomelag, outcomelagmean, datetimefeature]
    fulldf = pd.concat(fulllist, axis=1).dropna()

    splitdt = config['traintestsplit']['splitdt']

    outcome = fulldf.iloc[:, 0].copy()
    covariate = fulldf.iloc[:, 1:].copy()

    outcometrain = outcome.loc[outcome.index <= splitdt].copy()
    outcometest = outcome.loc[outcome.index > splitdt].copy()

    covariatetrain = covariate.loc[covariate.index <= splitdt].copy()
    covariatetest = covariate.loc[covariate.index > splitdt].copy()

    hyperparameters = {
        'max_depth': 5,
        'eta': .03,
        'silent': 1,
        'objective': 'reg:squarederror',
        'nthread': 5,
        'eval_metric': 'rmse',
        'num_boost_round': 2000,
        'early_stopping_rounds': 50
    }

    model = xgb.XGBRegressor(**hyperparameters)

    model.fit(covariatetrain, outcometrain, eval_metric='rmse')
    prediction = model.predict(covariate)

    predictiontest = model.predict(covariatetest)

    r2 = r2_score(y_true=outcometest, y_pred=predictiontest)



    prediction_val = go.Scatter(
        x=outcome.index,
        y=prediction,
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
                'x0': np.where(outcome.index > pd.to_datetime(splitdt))[0][0],
                'y0': 0.08,

                'x1': np.where(outcome.index > pd.to_datetime(splitdt))[0][0],
                'y1': 0.9,
                'line': {
                    'color': 'rgb(55, 128, 191)',
                    'width': 3}
            }
        ],

    )

    fig = dict(data=data, layout=layout)
    return fig


#dftest = dataprep.readdata('/Users/you/Desktop/quant/stockdataanalysis/raw/endprice.csv')


# @app.callback(Output('output-data-upload', 'children'),
#               [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename'),
#                State('upload-data', 'last_modified')])
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children


@app.callback(Output('timeseriesplot', 'figure'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        children = [
            parse_contents_getdf(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)][0]
        #print(children)
        return timeseriesfs(children)

    else:
        prediction_val = go.Scatter(
            x=[],
            y=[],
            name="Predicted Value",
            line=dict(color='#17BECF'),
            opacity=0.8)

        actual_val = go.Scatter(
            x=[],
            y=[],
            name="Actual Value",
            line=dict(color='#7F7F7F'),
            opacity=0.8)

        data = [prediction_val, actual_val]
        layout = dict(
            title=' ',

            xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                type='category'
            ),

        )

        empfig = dict(data=data, layout=layout)
        return(empfig)


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8000, debug=True)