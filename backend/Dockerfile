FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies with specific versions to avoid conflicts
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    flask-cors==4.0.0 \
    numpy==1.24.3 \
    opencv-python==4.8.1.78 \
    mediapipe==0.10.7 \
    tensorflow==2.13.0 \
    keras==2.13.1 \
    mysql-connector-python==8.1.0 \
    joblib==1.3.1 \
    pillow==10.0.0 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/dataset/collected_images

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/stats || exit 1

# Run the application
CMD ["python", "app.py"]