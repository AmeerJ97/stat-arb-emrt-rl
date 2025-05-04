FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt requirements-dev.txt ./
COPY src ./src
COPY docs ./docs
COPY references ./references

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[ui]"

EXPOSE 8501

CMD ["emrt-rl", "streamlit", "--server.address=0.0.0.0", "--server.port=8501"]
