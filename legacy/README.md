# mlapi

# Requirements
* [AWS CLI](https://aws.amazon.com/ko/cli/)
* [Python 3](https://www.python.org/downloads/)
* [AWS SAM CLI](https://aws.amazon.com/ko/serverless/sam/)

# setup
## Create S3 Bucket
```
aws s3 mb s3://REPLACE_WITH_YOUR_BUCKET_NAME
```

if "Unable to locate credentials" error occurs, configure it
```
aws configure
```

You may need to specify default region. e.g)
```
set AWS_DEFAULT_REGION=us-east-1
```

## Export your trained model and upload to S3
The SAM application expects a PyTorch model in [TorchScript](https://pytorch.org/docs/stable/jit.html#module-torch.jit) format to be saved to S3 along with a classes text file with the output class names.

```python
import boto3
s3 = boto3.resource('s3')
# replace 'mybucket' with the name of your S3 bucket
s3.meta.client.upload_file(tar_file, 'REPLACE_WITH_YOUR_BUCKET_NAME', 'fastai-models/lesson1/model.tar.gz')
```

# Reference
* [fastai: Deploying on AWS lambda](https://course19.fast.ai/deployment_aws_lambda.html)

----

# (Deprecated) fastapi 
# setup
```
python -m venv mlapi
mlapi\Scripts\activate
pip install uvicorn gunicorn fastapi pydantic scikit-learn pandas python-multipart
```
# run

```
uvicorn mlapi:app --reload
```

# deploy to AWS lambda
```
pip freeze > requirements.txt
pip install -t dependencies -r requirements.txt
(cd dependencies; zip ../aws_lambda_artifact.zip -r .)
zip aws_lambda_artifact.zip -u mlapi.py
```
upload aws_lambda_artifact.zip