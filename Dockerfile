FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PostgreSQL and building packages
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run Gunicorn with Uvicorn workers for production load balancing
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
