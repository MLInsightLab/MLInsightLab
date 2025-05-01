#!/bin/sh

# Pull the model deployment container image
printf "📦 Pulling the latest model container image...\n\n"
docker pull ghcr.io/mlinsightlab/mlinsightlab-model-container:latest

# Determine whether to use SSL
if [ -d "certs" ]; then
    printf "\n🔐 SSL certificates directory found. SSL deployment enabled.\n\n"
    USE_SSL="true"
else
    printf "\n⚠️  SSL certificates directory not found. Proceeding with non-SSL deployment.\n\n"
    USE_SSL="false"
fi

# Check for GPU availability
if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null; then
    printf "🖥️  GPU detected. Deploying with GPU support...\n\n"

    if [ "$USE_SSL" = "true" ]; then
        docker compose -f docker-compose.ssl.gpu.yaml up -d
    else
        docker compose -f docker-compose.nonssl.gpu.yaml up -d
    fi
else
    printf "🚫 No GPU detected or NVIDIA drivers missing. Deploying CPU-only version...\n\n"

    if [ "$USE_SSL" = "true" ]; then
        docker compose -f docker-compose.ssl.nongpu.yaml up -d
    else
        docker compose -f docker-compose.nonssl.nongpu.yaml up -d
    fi
fi
