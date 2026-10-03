FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY backend/pyproject.toml backend/pyproject.toml
COPY README.md README.md
COPY backend/app backend/app
RUN pip install --no-cache-dir -e "./backend[nlp]" \
 && pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY sample_data sample_data
COPY eval eval
COPY scripts scripts

WORKDIR /app/backend
EXPOSE 8000
# The embedding model (~90 MB) is downloaded on first start into the "models" volume.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
