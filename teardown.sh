#!/bin/sh

# Check if certificates directory exists
if test -d "certs"; then
    echo "SSL certificates directory found. Will tear down SSL version."
    USE_SSL="true"
else
    echo "SSL certificates directory not found. Will tear down non-SSL version."
    USE_SSL="false"
fi

# Check if nvidia-smi exists and detects a GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "GPU detected. Tearing down GPU version."

    # Tear down SSL or non-SSL version appropriately
    if [ "$USE_SSL" == "true" ]; then
        docker compose -f docker-compose.ssl.gpu.yaml down
    else
        docker compose -f docker-compose.nonssl.gpu.yaml down
    fi

# GPU not detected - tear down CPU only
else
    echo "No GPU detected or NVIDIA drivers not installed. Tearing down CPU version."

    # Tear down SSL or non-SSL version appropriately
    if [ "$USE_SSL" == "true" ]; then
        docker compose -f docker-compose.ssl.nongpu.yaml down
    else
        docker compose -f docker-compose.nonssl.nongpu.yaml down
    fi

fi
