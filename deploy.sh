#!/bin/sh

# Base compose file
BASE_COMPOSE="docker-compose.base.yaml"

# Pull the model deployment container image
printf "📦 Pulling the latest model container image...\n\n"
docker pull ghcr.io/mlinsightlab/mlinsightlab-model-container:latest

# Determine whether to use SSL
if [ -d "certs" ]; then
    printf "\n🔐 SSL certificates directory found. SSL deployment enabled.\n\n"
    NGINX_COMPOSE="docker-compose.ssl.yaml"
    
else
    printf "\n⚠️  SSL certificates directory not found. Proceeding with non-SSL deployment.\n\n"
    NGINX_COMPOSE="docker-compose.nonssl.yaml"
fi

# Check for GPU availability
if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null; then
    
    printf "🖥️  GPU detected. Deploying with GPU support...\n\n"
    GPU_COMPOSE="docker-compose.gpu.yaml"

# GPU not detected
else
    printf "🚫 No GPU detected or NVIDIA drivers missing. Deploying CPU-only version...\n\n"
    GPU_COMPOSE="docker-compose.nongpu.yaml"
fi

# Networks yaml
NETWORKS_COMPOSE="networks.yaml"

# Volumes yaml
VOLUMES_COMPOSE="volumes.yaml"

# Put all of the complete yaml files together
FILES="-f ${BASE_COMPOSE} -f ${NGINX_COMPOSE} -f ${GPU_COMPOSE} -f ${NETWORKS_COMPOSE} -f ${VOLUMES_COMPOSE}"

echo $FILES

# Pull the containers
docker compose ${FILES} pull

# Deploy
docker compose ${FILES} up -d
