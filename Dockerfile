FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python 3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MPLBACKEND=Agg \
    HF_HOME=/tmp/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    ENABLE_HF_MODEL_PREFETCH=1 \
    LLM_HUB_MODEL=ms-shamanth/recalltrace-investigator \
    LLM_BASE_MODEL=unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p plots

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
