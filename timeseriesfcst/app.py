import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.metrics import r2_score
import pickle
import numpy as np
import plotly
import plotly.graph_objs as go

#import timeseriesinput as tsinput
from xgbmodeling import xgbinput
#from xgbmodeling import xgbtraining
#import fulldata

storeid = "12"
Outputpath = "/Users/you/Desktop/timeseries/UltaDataAnalysis/Output/"

xgbinpt = xgbinput.getinputfromulta(storeid)


param = {
    'max_depth': 2,
    'eta': .05,
    'silent': 1,
    'objective': 'reg:squarederror',
    'nthread': 5,
    'eval_metric': 'rmse'
}


filename = Outputpath + str(storeid) + '.model'
xgbmod = pickle.load(open(filename, "rb"))

r2 = r2_score(y_true=xgbinpt.dtest.get_label(), y_pred=xgbmod.predict(xgbinpt.dtest))



prediction = go.Scatter(
    x=xgbinpt.traintestindex,
    y=xgbmod.predict(xgbinpt.dtraintest),
    name="Predicted Traffic",
    line=dict(color='#17BECF'),
    opacity=0.8)

actualtraffic = go.Scatter(
    x=xgbinpt.traintestindex,
    y=xgbinpt.dtraintest.get_label(),
    name="Actual Traffic",
    line=dict(color='#7F7F7F'),
    opacity=0.8)

data = [prediction, actualtraffic]


layout = dict(
    title='Prediction of hourly traffic' + " r2=" + str(round(r2, 2)),

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
            'x0': np.where(xgbinpt.traintestindex > pd.to_datetime('2018-09-01 00:00:00'))[0][0],
            'y0': 0.08,

            'x1': np.where(xgbinpt.traintestindex > pd.to_datetime('2018-09-01 00:00:00'))[0][0],
            'y1': 0.9,
            'line': {
                'color': 'rgb(55, 128, 191)',
                'width': 3}
        }
    ],

)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([
    dcc.Dropdown(id='my-id',
                 options=[{'label': '659 Freehold', 'value': '659'},
                          {'label': '317 Midland', 'value': '317'},
                          {'label': '634 Branson', 'value': '634'},
                          {'label': '590 Charlotte', 'value': '590'},
                          {'label': '1059 Forsyth', 'value': '1059'},
                          {'label': '449 Bozeman', 'value': '449'},
                          {'label': '662 Chicago', 'value': '662'},
                          {'label': '12 Hodgkins', 'value': '12'},
                          {'label': '737 Rockhill', 'value': '737'},
                          {'label': '619 Libson', 'value': '619'}],
                 value="659"),
    dcc.Graph(id='timeseriesplot'),

])

@app.callback(
    Output(component_id='timeseriesplot', component_property='figure'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    storeid = input_value
    Outputpath = "/Users/you/Desktop/timeseries/UltaDataAnalysis/Output/"

    xgbinpt = xgbinput.getinputfromulta(storeid)

    param = {
        'max_depth': 2,
        'eta': .05,
        'silent': 1,
        'objective': 'reg:squarederror',
        'nthread': 5,
        'eval_metric': 'rmse'
    }

    filename = Outputpath + str(storeid) + '.model'
    xgbmod = pickle.load(open(filename, "rb"))

    r2 = r2_score(y_true=xgbinpt.dtest.get_label(), y_pred=xgbmod.predict(xgbinpt.dtest))

    prediction = go.Scatter(
        x=xgbinpt.traintestindex,
        y=xgbmod.predict(xgbinpt.dtraintest),
        name="Predicted Traffic",
        line=dict(color='#17BECF'),
        opacity=0.8)

    actualtraffic = go.Scatter(
        x=xgbinpt.traintestindex,
        y=xgbinpt.dtraintest.get_label(),
        name="Actual Traffic",
        line=dict(color='#7F7F7F'),
        opacity=0.8)

    data = [prediction, actualtraffic]

    layout = dict(
        title='Prediction of hourly traffic in store '+str(storeid) + ", r2=" + str(round(r2, 2)),

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
                'x0': np.where(xgbinpt.traintestindex > pd.to_datetime('2018-09-01 00:00:00'))[0][0],
                'y0': 0.08,

                'x1': np.where(xgbinpt.traintestindex > pd.to_datetime('2018-09-01 00:00:00'))[0][0],
                'y1': 0.9,
                'line': {
                    'color': 'rgb(55, 128, 191)',
                    'width': 3}
            }
        ],

    )

    fig = dict(data=data, layout=layout)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
