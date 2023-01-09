from fastapi import FastAPI
import add_router
import uvicorn

app = FastAPI()

app.include_router(add_router.ncf)
app.include_router(add_router.nfm)

@app.get('/')
def home():
    return {'msg' : 'Main'}

if __name__ == "__main__":
    uvicorn.run("router_example:app", host='0.0.0.0', port=8000, reload=True)