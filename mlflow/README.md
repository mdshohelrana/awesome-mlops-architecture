bash```
mlflow ui --port 6000
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
python quick_start.py

mlflow ui

mlflow models serve --model-uri runs:/d3cf2b4ac67748f38683292005447e51/model --no-conda --port 5001

```