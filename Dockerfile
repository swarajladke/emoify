# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set permissions for face storage
RUN chmod 777 static

# Hugging Face Spaces expects the app on port 7860
ENV PORT=7860
EXPOSE 7860

# Run the application with Gunicorn
CMD ["gunicorn", "--timeout", "120", "--workers", "1", "-b", "0.0.0.0:7860", "app:app"]
