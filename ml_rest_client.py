import json
import requests

url = 'http://127.0.0.1:8002/model'

request_data = json.dumps({'age':42, 'salary':50000})
response = requests.post(url, request_data)
print(response.text)