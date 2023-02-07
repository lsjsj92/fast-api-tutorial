from fastapi import APIRouter
import torch

from packages.config import DataInput, PredictOutput
from packages.config import ProjectConfig


# Project config 설정
project_config = ProjectConfig('ncf')
# 모델 가져오기
model = project_config.load_model()
model.eval()

ncf = APIRouter(prefix='/ncf')

# router 마다 경로 설정
@ncf.get('/', tags=['ncf'])
async def start_ncf():
    return {'msg' : 'Here is NCF'}

@ncf.post('/predict', tags=['ncf'], response_model=PredictOutput)
async def ncf_predict(data_request: DataInput):
    user_id = data_request.user_id
    item_id = data_request.movie_id
    predict = model(torch.tensor( [[user_id, item_id]] ))
    prob, prediction = predict, int(( predict > project_config.threshold ).float() * 1) 
    return {'prob' : prob, 'prediction' : prediction}