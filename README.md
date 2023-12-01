# mlapi

# setup
```
python -m venv mlapi
mlapi\Scripts\activate
pip install uvicorn gunicorn fastapi pydantic scikit-learn pandas
```

# install

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
