from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import os, uuid

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

MODEL = './model/best.torchscript'

@app.post("/detect")
async def detect(file: UploadFile):
    UPLOAD_DIR = './images'
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(os.path.abspath('.'))

    content = await file.read()
    filename = f'{str(uuid.uuid4())}.jpg'
    filename = os.path.join(UPLOAD_DIR, filename) 
    
    # image save test. no need to save
    with open(filename, 'wb') as fp:
        fp.write(content)
    
    res = []
    try:
        res = inference(MODEL, filename)
    except Exception as ex:
        print(ex)
    return {'bbox': res}

def inference(model, image):
    import torch
    print (torch.__version__)

    torch.load(model)
    # TODO: load and run
    return [[120,120,120,120]]

