FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pyopengl
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV OPEN3D_DISABLE_WEB_VISUALIZER=1
COPY reconstruct.py .
CMD ["uvicorn", "reconstruct:app", "--host", "0.0.0.0", "--port", "8001"]
