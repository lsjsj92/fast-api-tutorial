import requests
import random

headers = {
    "Content-type": "application/json",
    "accept": "application/json"
}


url = "http://localhost:8088/ncf/predict"
params={
    'user_id': 201,
    'movie_id' : 100,
    'gender' : 0,
    'age' : 5,
    'occupation' : 8,
    'genre' : 4
}

res = requests.post(url, json=params, headers=headers)

print("status : ", res.status_code)
print(res.json())


