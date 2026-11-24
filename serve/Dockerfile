# Small inference image: onnxruntime + FastAPI, no PyTorch. Build from the repo root:
#   docker build -f serve/Dockerfile -t clearscan . && docker run -p 8000:8000 clearscan
FROM python:3.13-slim
WORKDIR /app
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt
COPY serve/app.py serve/app.py
COPY models/clearscan_int8.onnx models/serving.json models/
COPY MODEL_CARD.md .
ENV CLEARSCAN_HOME=/app CLEARSCAN_THREADS=1
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
