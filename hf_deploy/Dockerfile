# Ensure we use an official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install all dependencies (includes uvicorn, opencv-headless, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set permissions for Hugging Face Spaces (runs as non-root user '1000')
RUN chown -R 1000:1000 /app
USER 1000

# Expose the default FastAPI port for Hugging Face (7860)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--limit-concurrency", "4"]
