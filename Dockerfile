# ML Search Service Dockerfile
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgflags-dev \
    libsnappy-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/data/faiss_index

# Pre-download models during build (optional - reduces startup time)
ARG SKIP_MODEL_DOWNLOAD=false
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
        python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" || echo "YOLO model download skipped"; \
        python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', show_log=False)" || echo "OCR model download skipped"; \
    else \
        echo "Skipping model pre-download"; \
    fi

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
