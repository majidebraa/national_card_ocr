# syntax=docker/dockerfile:1.7
FROM python:3.15-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # One OpenMP/OpenCV thread per request: parallelism comes from IDCARD_MAX_CONCURRENCY,
    # which keeps CPU use predictable on small hosts.
    OMP_THREAD_LIMIT=1 \
    IDCARD_CV_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-fas tesseract-ocr-ara tesseract-ocr-eng fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --create-home --home-dir /home/app app \
    && mkdir /data && chown app:app /data

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

ENTRYPOINT ["idcard-ocr"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
