# Use a public mirror of Docker Hub to avoid 401/rate-limit issues
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Recommended runtime flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py ./app.py

# Override these at deploy time (don’t hardcode secrets in the image)
ENV API_KEY=changeme CORS_ORIGIN="*" PORT=8000

EXPOSE 8000

# Run with uvicorn (app.py defines `app = FastAPI()`)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
