import pymongo
import time

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["quantaxis"]
# print(myclient.list_database_names())

# print(mydb.list_collection_names())
quant = mydb['stock_day']
myquery = {"code": "000002"}

mydoc = quant.find(myquery)
for x in mydoc:
    datatest = myclient['mydatabase']
    testcol = datatest['000002']
    testcol.insert_one(x)
    time.sleep(1)

if __name__ == '__main__':
    test.run(debug=True)
