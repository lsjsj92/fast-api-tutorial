from fastapi import APIRouter

ncf = APIRouter(prefix='/ncf')
nfm = APIRouter(prefix='/nfm')

@ncf.get('/', tags=['ncf'])
async def start_ncf():
    return {'msg' : 'Here is NCF'}

@ncf.get('/model', tags=['ncf'])
async def start_ncf():
    return {'msg' : 'Here is NCF model'}

@nfm.get('/', tags=['nfm'])
async def start_nfm():
    return {'msg' : 'Here is NFM'}


