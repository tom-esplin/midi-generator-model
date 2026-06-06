FROM python:3.12-slim

WORKDIR /app

ENV FLASK_APP=backend.app \
    PORT=7860 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY models models
COPY tokenization tokenization

EXPOSE 7860

CMD ["sh", "-c", "flask run --host 0.0.0.0 --port ${PORT}"]
