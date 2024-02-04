import json
import torch

try:
    import unzip_requirements
except ImportError:
    pass

import json
from io import BytesIO
import time
import os
import base64

import boto3
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

s3 = boto3.client("s3")
bucket = "soltware.test"
key = "v1/best.torchscript"

def download_model(s3, bucket, key):
    file_name = os.path.basename(key)
    print ('file_name', file_name)
    s3.download_file(bucket, key, file_name)


def load_model(s3, bucket):
  response = s3.get_object(Bucket=bucket, Key=key)
  #state = torch.load(BytesIO(response["Body"].read()))
  #model.load_state_dict(state)
  #model.eval()

  bytes_array = BytesIO(response["Body"].read())
  model = torch.jit.load(bytes_array, map_location=torch.device('cpu')).eval()
  return model

model = load_model(s3, bucket)

classes = np.array([
  'Tomato Healthy',
  'Strawberry Healthy',
  'Lettuce Healthy',
  'Strawberry Ashy Mold',
  'Strawberry White Powdery Mildew',
  'Lettuce Bacterial Head Rot',
  'Lettuce Bacterial Wilt',
  'Tomato Leaf Mold',
  'Tomato Yellow Leaf Curl Virus',
])

def lambda_handler(event, context):
    '''
    if event.get("source") in ["aws.events", "serverless-plugin-warmup"]:
        print('Lambda is warm!')
        return {}
    '''

    data = json.loads(event["body"])
    print("data keys:", data.keys())
    image = data["image"]
    response = predict(input_fn_stream(image), model)
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }

def input_fn_stream(image):
    image = image[image.find(",")+1:]
    dec = base64.b64decode(image + "===")
    byte_array = BytesIO(dec)

    im = Image.open(byte_array).resize((640,640))
    im = im.convert("RGB")

    #https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e#prepare_the_input
    # "We do not need Alpha channel in the image for YOLOv8 predictions. Let's remove it"
    input = np.array(im)
    input = input.transpose(2,0,1)
    input = input.reshape(1,3,640,640)
    input = input/255.0
    return torch.Tensor(input)

def predict(img_tensor, model):
  predict_values = model(img_tensor)
  print(predict_values[0].shape)
  print('predict_values[0]', predict_values[0])
  preds = F.softmax(predict_values, dim=1)
  conf_score, indx = torch.max(preds, dim=1)
  conf_score = conf_score.cpu().numpy()
  indx = indx.cpu().numpy()
  predict_class = classes[indx]
  response = {}
  response['class'] = str(predict_class)
  response['confidence'] = str(conf_score)
  return response



