# Smart City Computer Vision - Docker Configuration
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models/garbage models/helmet models/traffic outputs logs

# Set permissions
RUN chmod +x *.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import ultralytics; print('OK')" || exit 1

# Default command
CMD ["python", "setup.py"]

# Labels for metadata
LABEL maintainer="Abdul Rafay <abdrafdev@gmail.com>"
LABEL description="Smart City Computer Vision - AI-powered urban monitoring"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/abdrafdev/smartcity-computer-vision"