FROM python:3.7

WORKDIR /app

COPY iris.csv ./iris.csv
COPY train.py ./train.py
COPY predict.py ./predict.py

RUN pip install pandas scikit-learn joblib

RUN python train.py

CMD ["python", "predict.py"]
