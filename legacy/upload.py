import boto3
s3 = boto3.resource('s3')
tar_file = 'model/model.tar.gz'
BUCKET='soltware.test'
s3.meta.client.upload_file(tar_file, BUCKET, 'v1/model.tar.gz')