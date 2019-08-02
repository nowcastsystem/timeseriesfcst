#!flask/bin/python
from flask import Flask, jsonify
from flask import request
from flask import abort
import tushare as ts
import pandas as ps

app = Flask(__name__)


@app.route('/', methods=['POST'])
def fetch_data():
    if not request.json or not 'name' in request.json :
        abort(400)

    datatest = ts.get_hist_data(request.json['name'])
    datatest = datatest.sort_index(ascending=True)
    datatest = datatest.reset_index()
    datatest['_id'] = datatest.index + 1
    dataset = datatest.to_json(orient='records')
    return dataset


if __name__ == '__main__':
    app.run(debug=True)
