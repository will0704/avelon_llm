# CPU-only image for Cloud Run. The three model sets are downloaded at build time,
# not at runtime — Cloud Run's filesystem is ephemeral, so anything fetched on first
# use is re-fetched on every cold start (~800MB).

FROM python:3.11-slim-bookworm

# Pinned to bookworm: trixie dropped libgl1-mesa-glx, which the previous base image
# resolved to and could no longer install. libgl1 + libglib2.0-0 are opencv's runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# EasyOCR and transformers read these. insightface has no equivalent env var — it
# always uses $HOME/.insightface — so the container has to keep running as root
# for the baked-in weights to be found at runtime.
ENV HF_HOME=/app/.cache/huggingface \
    EASYOCR_MODULE_PATH=/app/.cache/easyocr \
    PYTHONUNBUFFERED=1

# CPU wheels on their own layer, before requirements.txt. The default PyPI torch is
# the CUDA build — several GB of GPU libraries this service never uses.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# insightface ships as a source distribution and compiles a Cython extension against
# numpy's headers, so both have to already be present when pip reaches it.
RUN pip install --no-cache-dir numpy Cython

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Warm the caches into an image layer.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=0, det_size=(640, 640))" \
    && python -c "import easyocr; easyocr.Reader(['en', 'tl'], gpu=False, verbose=False)" \
    && python -c "from transformers import pipeline; pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')"

EXPOSE 8000

# Cloud Run injects PORT; the default keeps a plain `docker run` on 8000.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
