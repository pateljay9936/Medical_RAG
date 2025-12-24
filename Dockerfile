FROM python:3.10-slim

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt setup.py ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directories
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface /app/.cache/torch

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Set the port for the app
ENV PORT=7860

CMD ["python", "app.py"]
