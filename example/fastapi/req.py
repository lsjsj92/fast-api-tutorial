import requests
import random

url = "http://127.0.0.1:8081/"
params={'something':f'{random.randint(0, 100)}'}

res = requests.post(url, params=params)

print("status : ", res.status_code)
print(res.json())
