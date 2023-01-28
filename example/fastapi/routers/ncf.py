from fastapi import APIRouter

ncf = APIRouter(prefix='/ncf')

@ncf.get('/', tags=['ncf'])
async def start_ncf():
    return {'msg' : 'Here is NCF'}

@ncf.get('/model', tags=['ncf'])
async def ncf_model():
    return {'msg' : 'Here is NCF model'}