FROM python:3.11-slim

# Install system dependencies for OpenCV, audio codecs, and headless operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with UID 1000 for Hugging Face Spaces non-root runtime
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    PORT=7860

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository files and ensure ownership by UID 1000
COPY --chown=user:user . /app

# Switch to non-root user
USER user

# Expose default Hugging Face Spaces port 7860
EXPOSE 7860

# Run uvicorn bound to 0.0.0.0 and port 7860 (or $PORT)
CMD ["sh", "-c", "python3 -m uvicorn dashboard.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
