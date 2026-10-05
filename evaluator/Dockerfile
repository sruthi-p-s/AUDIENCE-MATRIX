FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=api \
    API_PORT=5000 \
    METRICS_OUTPUT=/app/models/metrics.json

COPY evaluator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluator/evaluate.py /app/evaluator/evaluate.py

CMD ["python", "evaluator/evaluate.py"]
