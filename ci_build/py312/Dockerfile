FROM ubuntu:24.04

ARG PB_REL="https://github.com/protocolbuffers/protobuf/releases"
ARG PB_VERSION="29.5"

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    wget \
    unzip \
    build-essential \
    rsync \
    sudo \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Protoc Compiler from zip
RUN cd tmp \
    && curl -LO ${PB_REL}/download/v${PB_VERSION}/protoc-${PB_VERSION}-linux-x86_64.zip \
    && unzip protoc-${PB_VERSION}-linux-x86_64.zip -d /usr/local \
    && rm protoc-${PB_VERSION}-linux-x86_64.zip

ENV PATH="/usr/local/bin:${PATH}"

# Change work directory
WORKDIR /workspace

CMD ["bash"]
