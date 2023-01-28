from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field

app = FastAPI()

class DataInput(BaseModel):
    name: str

class PredictOutput(BaseModel):
    prob:float
    prediction:int


@app.get("/")
def home():
    return {"Hello": "GET"}

@app.post("/")
def home_post(data_request: DataInput):
    return {"Hello": "POST", "msg" : data_request.name}

@app.post("/pydantic", response_model=PredictOutput)
def pydantic_post(data_request: DataInput):
    return {"prob": 0.1, "prediction" : 0}