FROM python:latest

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN python train.py

CMD ["python", "app.py"]