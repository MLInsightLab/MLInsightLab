#!/bin/sh

# Base compose file
BASE_COMPOSE="docker-compose.base.yaml"

# Determine whether to tear down SSL version
if [ -d "certs" ]; then
    printf "\n🔐 SSL certificates directory found. SSL teardown initiated.\n\n"
    NGINX_COMPOSE="docker-compose.ssl.yaml"
else
    printf "\n⚠️  SSL certificates directory not found. Proceeding with non-SSL teardown.\n\n"
    NGINX_COMPOSE="docker-compose.nonssl.yaml"
fi

# Check for GPU availability
if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null; then    
    printf "🖥️  GPU detected. Tearing down GPU version...\n\n"
    GPU_COMPOSE="docker-compose.gpu.yaml"

# GPU not detected - tear down CPU only
else
    printf "🚫 No GPU detected or NVIDIA drivers missing. Tearing down CPU-only version...\n\n"
    GPU_COMPOSE="docker-compose.nongpu.yaml"
fi

# Tear down the services
docker compose -f ${BASE_COMPOSE} -f ${NGINX_COMPOSE} -f ${GPU_COMPOSE} down
