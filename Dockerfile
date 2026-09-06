# Use Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    NLTK_DATA=/home/user/nltk_data

# Install system dependencies (fonts for ReportLab, build tools, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    fonts-dejavu \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000 required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Upgrade pip and install CPU-only PyTorch (reduces image size from 2.5GB to ~180MB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download required NLTK datasets so container does not require internet at runtime
RUN mkdir -p /home/user/nltk_data && \
    python -c "import nltk; \
    nltk.download('punkt', download_dir='/home/user/nltk_data', quiet=True); \
    nltk.download('punkt_tab', download_dir='/home/user/nltk_data', quiet=True); \
    nltk.download('stopwords', download_dir='/home/user/nltk_data', quiet=True); \
    nltk.download('wordnet', download_dir='/home/user/nltk_data', quiet=True); \
    nltk.download('averaged_perceptron_tagger', download_dir='/home/user/nltk_data', quiet=True); \
    nltk.download('vader_lexicon', download_dir='/home/user/nltk_data', quiet=True)"

# Copy the rest of the application files
COPY --chown=user:user . .

# Ensure upload, data, and output directories exist with proper permissions
RUN mkdir -p /home/user/app/data/raw /home/user/app/data/processed /home/user/app/output

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Launch Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
