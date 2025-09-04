FROM python:3.11.13-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      wget \
      jq \
      zip \
      sudo \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY .env /workspace/.env

COPY pyproject.toml setup.cfg* /workspace/

COPY src/ /workspace/src/

RUN pip install --no-cache-dir setuptools wheel \
 && pip install --no-cache-dir .

VOLUME ["/workspace/data", "/workspace/logs"]

CMD ["python", "src/manager.py"]
