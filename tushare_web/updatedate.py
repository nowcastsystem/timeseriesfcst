import datetime
import pymongo
import tushare as ts
import json

date = str(datetime.date.today())
newdata = ts.get_hist_data('600848', start=date, end=date)
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]

mdtest = mydb[str(name)]
newdata = newdata.reset_index()
newdata['_id'] = mdtest.count() + 1
newdata = newdata.to_json(orient='records')
newdata = json.loads(newdata)
x = mdtest.insert_one(newdata[0])
