# Operations

## Sizing for small CPU-only hosts

* Start with `IDCARD_MAX_CONCURRENCY` equal to the number of cores, at most 2 on a tiny VM.
  Each in-flight request holds a decoded photo plus working copies (tens of MB for a 12 MP image).
* Lower `IDCARD_MAX_PIXELS` (default 40 MP) and `IDCARD_MAX_UPLOAD_BYTES` (default 8 MiB) if your
  clients downscale before upload; 1600 px on the long edge is plenty.
* Keep `OMP_THREAD_LIMIT=1` and `IDCARD_CV_THREADS=1` (the image sets both) and scale with replicas.
* Typical latency in the synthetic benchmark is 0.3 s at p50 and under 1 s at p95 per image on one core.
  Measure on your hardware with `idcard-ocr bench`.

## Configuration reference

All settings are environment variables (see `src/idcard_ocr/config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `IDCARD_API_KEYS` | none | comma-separated API keys (required unless anonymous) |
| `IDCARD_ALLOW_ANONYMOUS` | `0` | development only |
| `IDCARD_STORE_DIR` / `IDCARD_STORE_KEYS` | none | enable retention; first key encrypts, all decrypt |
| `IDCARD_RETENTION_DAYS` | `30` | retention period |
| `IDCARD_MAX_CONCURRENCY` | `min(2, cpus)` | parallel images |
| `IDCARD_QUEUE_TIMEOUT` | `10` | seconds a request may wait for a slot before `429` |
| `IDCARD_REQUEST_TIMEOUT` | `90` | seconds before `504` |
| `IDCARD_MAX_UPLOAD_BYTES` | `8388608` | upload cap |
| `IDCARD_MAX_PIXELS` | `40000000` | decoded-pixel cap |
| `IDCARD_QUALITY_MODE` | `enforce` | `warn` reports quality issues without rejecting |
| `IDCARD_MIN_SHARPNESS` | `45` | Laplacian-variance floor on the rectified card |
| `IDCARD_MIN_BRIGHTNESS` / `IDCARD_MAX_BRIGHTNESS` | `50` / `225` | mean grey bounds |
| `IDCARD_MAX_GLARE_RATIO` | `0.05` | share of field regions allowed to be blown out |
| `IDCARD_MIN_SOURCE_WIDTH` | `520` | minimum card width in source pixels |
| `IDCARD_REVIEW_CONFIDENCE` | `0.60` | below this a field is flagged |
| `IDCARD_TRUST_NAMES` | `0` | allow names to skip review (not recommended) |
| `IDCARD_EXTRACT_PORTRAIT` | `1` | crop the face photo (YuNet, ~40 ms/card); `0` disables |
| `IDCARD_OCR_LANG` / `IDCARD_OCR_ALT_LANG` / `IDCARD_OCR_LATIN_LANG` | `fas` / `ara` / `eng` | Tesseract languages (alt empty disables) |
| `IDCARD_TESSDATA_DIR` | system | directory with `*.traineddata` |
| `IDCARD_OCR_TIMEOUT` | `20` | seconds per Tesseract call |
| `IDCARD_LAYOUT_DIR` | none | extra/overriding layout JSON files |

## Retention and keys

```bash
idcard-ocr gen-key                       # new 256-bit key
# rotate: IDCARD_STORE_KEYS="<new>,<old>"  (new data uses the first; old data still readable)
idcard-ocr purge                         # run daily from cron to delete expired images
```

Back up the key ring separately from the data volume. Losing every key makes the images
unrecoverable by design.

## Deployment checklist

* TLS terminated by a reverse proxy; do not expose the container port directly.
* Per-client rate limiting at the proxy (the app only bounds total concurrency).
* Encrypted disk or volume under `IDCARD_STORE_DIR` in addition to application encryption.
* Log shipping: application logs contain method, path, status and duration only. Keep it that way.
* Run `/readyz` as the readiness probe and `/healthz` as liveness.
* Pin the image by digest; Dependabot is configured for pip, Docker and Actions.
