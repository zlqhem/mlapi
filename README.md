# mlapi

# Requirements
* [AWS CLI](https://aws.amazon.com/ko/cli/)
* [Python 3](https://www.python.org/downloads/)
* [AWS SAM CLI](https://aws.amazon.com/ko/serverless/sam/)
* Docker CLI

# Create an ECR repository
```
aws ecr create-repository --repository-name lambda-pytorch-example --image-scanning-configuration scanOnPush=true --region <REGION>
```

# Register docker to ECR
```
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
```

# Deploying the application
```
sam build && sam deploy –-guided
```




# Reference
* [aws:using-container-images-to-run-pytorch-models-in-aws-lambda](https://aws.amazon.com/ko/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/)


