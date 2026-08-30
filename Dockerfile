FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and multimedia
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements with CPU wheels for torch to keep container lean and fast
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
EXPOSE 7860
EXPOSE 8000
EXPOSE 10000

ENV OMP_NUM_THREADS=1
ENV MALLOC_ARENA_MAX=2

# Command to run on port 7860 (Hugging Face default) or 8000 or Render $PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
