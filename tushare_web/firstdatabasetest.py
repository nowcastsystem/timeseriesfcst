import pymongo
import urllib.request
import json

name = 600848
textmod = {"name": str(name)}
# json input
textmod = json.dumps(textmod).encode(encoding='utf-8')
# regular input
# textmod = parse.urlencode(textmod).encode(encoding='utf-8')
url = 'http://127.0.0.1:5000/'
headers = {'Content-Type': 'application/json'}
req = urllib.request.Request(url, textmod, headers)
response = urllib.request.urlopen(req)

result = response.read()
result = result.decode(encoding='utf-8')
result = json.loads(result)
# print(result.decode(encoding='utf-8'))
# print(result)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]

mdtest = mydb[str(name)]
x = mdtest.insert_many(result)
doc = mdtest.find()
for i in doc:
    print(i)
