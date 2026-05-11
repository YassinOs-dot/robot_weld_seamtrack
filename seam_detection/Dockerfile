
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY detect.py .
CMD ["uvicorn", "detect:app", "--host", "0.0.0.0", "--port", "8002"]
