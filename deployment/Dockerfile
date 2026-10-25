# Base Build Layer using Python 3.11 Slim Image
FROM python:3.11-slim

# Set system variables for optimal Python execution inside container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/app

# Create work directory
WORKDIR ${WORKDIR}

# Install system dependencies if required (clean build setup)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first to leverage Docker cache layers
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire workspace/project structure to the container
COPY . .

# Expose target network routing ports dynamically occupied by services
EXPOSE 8000
EXPOSE 8501

# Default runtime placeholder (Overridden cleanly inside docker-compose)
CMD ["python"]
