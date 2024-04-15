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

def load_model(s3, bucket):
  response = s3.get_object(Bucket=bucket, Key=key)
  #state = torch.load(BytesIO(response["Body"].read()))
  #model.load_state_dict(state)
  #model.eval()

  bytes_array = BytesIO(response["Body"].read())
  model = torch.jit.load(bytes_array, map_location=torch.device('cpu')).eval()
  return model

model = load_model(s3, bucket)


input_width=640
input_height=640
conf_threshold=0.3
iou_threshold=0.5

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
    try:
        data = json.loads(event["body"])
        print("data keys:", data.keys())
        image = data["image"]
        img_width = int(data["width"])
        img_height = int(data["height"])

        response = predict(input_fn_stream(image), model, img_width, img_height)

        return {
            'statusCode': 200,
            'headers': {
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Origin": "*", # Required for CORS support to work
                "Access-Control-Allow-Methods": "*",
                "Accept": "*/*",
                "Access-Control-Allow-Credentials": "true", # Required for cookies, authorization headers with HTTPS
            },
            'body': json.dumps(response)
        }
    except Exception as ex:
        return {
            'statusCode': 500,
            'msg': repr(ex),
            'headers': {
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Origin": "*", # Required for CORS support to work
                "Access-Control-Allow-Methods": "*",
                "Accept": "*/*",
                "Access-Control-Allow-Credentials": "true", # Required for cookies, authorization headers with HTTPS
            }
        }

def input_fn_stream(image):
    image = image[image.find(",")+1:]
    dec = base64.b64decode(image + "===")
    byte_array = BytesIO(dec)

    # FIXME: get model input shape.
    input_widh = 640
    input_height = 640
    im = Image.open(byte_array).resize((input_widh,input_height))
    im = im.convert("RGB")

    #https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e#prepare_the_input
    # "We do not need Alpha channel in the image for YOLOv8 predictions. Let's remove it"
    input = np.array(im)
    input = input.transpose(2,0,1)
    input = input.reshape(1,3,input_widh,input_height)
    input = input/255.0
    return torch.Tensor(input)

'''
{'predictions':
  [{'x': 1012.0, 'y': 593.5, 'width': 406.0, 'height': 443.0, 'confidence': 0.7369905710220337, 'class': 'Paper',
   'image_path': 'example.jpg', 'prediction_type': 'ObjectDetectionModel'}],
  'image': {'width': 1436, 'height': 956}}
'''
#https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/blob/main/yolov8%2FYOLOv8.py#L66
def predict(img_tensor, model, img_width, img_height):
  outputs = model(img_tensor)
  boxes, scores, class_ids = process_output(outputs, img_width, img_height)
  for box, score, class_id in zip(boxes, scores, class_ids):
    print(box, score, class_id)

  response = {}
  #[x,y,x,y]
  response['boxes'] = str(boxes)
  response['scores'] = str(scores)
  response['class_ids'] = str(class_ids)
  response['labels'] = str([classes[id] for id in class_ids])
  print(response)
  return response

def process_output(output, img_width, img_height):
    #// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    output = output.numpy()
    predictions = np.squeeze(output[0]).T

    print('predictions.shape', predictions.shape)
    print('pred[0] box', predictions[0][:4])
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], []

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions, img_width, img_height)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    # indices = nms(boxes, scores, self.iou_threshold)
    indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)

    return boxes[indices], scores[indices], class_ids[indices]

def extract_boxes(predictions, img_width, img_height):
    # Extract boxes from predictions
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes, img_width, img_height)

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    return boxes

def rescale_boxes(boxes, img_width, img_height):

    # Rescale boxes to original image dimensions
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_width, img_height, img_width, img_height])
    return boxes

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

