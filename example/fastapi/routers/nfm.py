from fastapi import APIRouter

nfm = APIRouter(prefix='/nfm')

@nfm.get('/', tags=['nfm'])
async def start_nfm():
    return {'msg' : 'Here is NFM'}

@nfm.get('/model', tags=['nfm'])
async def nfm_model():
    return {'msg' : 'Here is NFM model'}
