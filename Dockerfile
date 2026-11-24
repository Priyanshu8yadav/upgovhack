FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Bal-Setu port
EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8080"]
