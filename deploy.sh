#!/bin/sh

# Check if certificates directory exists
if test -d "certs"; then
    echo "SSL certificates directory found. Will deploy SSL version."
    USE_SSL="true"
else
    echo "SSL certificates directory not found. Will deploy non-SSL version."
    USE_SSL="false"
fi

# Check if nvidia-smi exists and detects a GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "GPU detected. Deploying with GPU support."
    
    # Deploy SSL or non-SSL version appropriately
    if [ "$USE_SSL" = "true" ]; then
        docker compose -f docker-compose.ssl.gpu.yaml up -d 
    else
        docker compose -f docker-compose.nonssl.gpu.yaml up -d 
    fi

# GPU not detected - deploy CPU only
else
    echo "No GPU Detected or NVIDIA drivers not installed. Deploying CPU only."
    
    # Tear down SSL or non-SSL version appropriately
    if [ "$USE_SSL" = "true" ]; then
        docker compose -f docker-compose.ssl.nongpu.yaml up -d
    else
        docker compose -f docker-compose.nonssl.nongpu.yaml up -d
    fi

fi
