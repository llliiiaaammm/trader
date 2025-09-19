FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./app.py
ENV API_KEY=changeme CORS_ORIGIN=* PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
