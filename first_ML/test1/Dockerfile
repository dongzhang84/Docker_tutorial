FROM python:latest

WORKDIR /app

COPY . /app

RUN pip install pandas scikit-learn joblib

RUN python train.py

CMD ["python", "predict.py"]
