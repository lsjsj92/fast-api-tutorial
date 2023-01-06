from fastapi import FastAPI
import time
import asyncio


async def some_library(num: int):
    s = 0
    for i in range(num):
        print(" i : ", i)
        await asyncio.sleep(1)
        s += 1
    return s

app = FastAPI()

@app.get('/')
async def read_results():
    s1 = await some_library(5)
    print("somthing ")
    print(" s1 : ", s1)
    return {'data' : 'data', 's1':s1}


