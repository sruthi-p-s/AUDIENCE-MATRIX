FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MODELS_DIR=/app/models \
    DATA_DIR=/app/data \
    CATALOG_PATH=/app/api/recommendation_catalog.json \
    METRICS_PATH=/app/models/metrics.json

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ /app/api/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "30", "--chdir", "/app/api", "app:app"]
