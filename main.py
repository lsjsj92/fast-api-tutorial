import argparse
import os

from fastapi import FastAPI
from packages import ncf, nfm
from packages import FastAPIRunner

app = FastAPI()

app.include_router(ncf)
app.include_router(nfm)

@app.get('/')
def read_results():
    return {'msg' : 'Main'}
    
if __name__ == "__main__":
    # python main.py --host 127.0.0.1 --port 8000
    parser = argparse.ArgumentParser()
    parser.add_argument('--host')
    parser.add_argument('--port')
    args = parser.parse_args()
    api = FastAPIRunner(args)
    api.run()
    