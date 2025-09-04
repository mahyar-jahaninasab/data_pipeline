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

COPY pyproject.toml setup.cfg* /workspace/

RUN pip install --no-cache-dir setuptools wheel \
 && pip install --no-cache-dir .

COPY src/ /workspace/src/
RUN echo "devcontainer" >> /root/.pipeline

SHELL ["/bin/bash", "-lc"]
