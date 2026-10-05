FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY trainer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY trainer/train.py /app/trainer/train.py

CMD ["python", "trainer/train.py"]
