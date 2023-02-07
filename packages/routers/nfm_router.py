from fastapi import APIRouter
import torch

from packages.config import DataInput, PredictOutput
from packages.config import ProjectConfig


# Project config 설정
project_config = ProjectConfig('nfm')
# 모델 가져오기
model = project_config.load_model()
model.eval()

nfm = APIRouter(prefix='/nfm')

@nfm.get('/', tags=['nfm'])
async def start_nfm():
    return {'msg' : 'Here is NFM'}

@nfm.post('/predict', tags=['nfm'], response_model=PredictOutput)
async def nfm_predict(data_request: DataInput):
    user_id = data_request.user_id
    item_id = data_request.movie_id
    gender = data_request.gender
    age = data_request.age
    occupation = data_request.occupation
    genre = data_request.genre
    predict = model(torch.tensor( [[user_id, item_id]] ), torch.tensor( [[gender, age, occupation]] ), torch.tensor( [[genre]] ) )
    prob, prediction = predict, int(( predict > project_config.threshold ).float() * 1) 
    return {'prob' : prob, 'prediction' : prediction}
