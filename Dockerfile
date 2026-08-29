FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and multimedia
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 7860
EXPOSE 8000

# Command to run on port 7860 (Hugging Face default) or 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
