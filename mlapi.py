from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# TODO: receive img blob from a client
class Request(BaseModel):
    img_path: str

@app.get('/')
async def main(req:Request):
    # torch script model inference here
    #yhat = model.predict()
    yhat = [[120,120,120,120]]
    return {"bbox": yhat}