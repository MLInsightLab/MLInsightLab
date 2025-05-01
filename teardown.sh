#!/bin/sh

# Determine whether to tear down SSL version
if [ -d "certs" ]; then
    printf "\n🔐 SSL certificates directory found. SSL teardown initiated.\n\n"
    USE_SSL="true"
else
    printf "\n⚠️  SSL certificates directory not found. Proceeding with non-SSL teardown.\n\n"
    USE_SSL="false"
fi

# Check for GPU availability
if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null; then
    printf "🖥️  GPU detected. Tearing down GPU version...\n\n"

    # Tear down SSL or non-SSL version appropriately
    if [ "$USE_SSL" = "true" ]; then
        docker compose -f docker-compose.ssl.gpu.yaml down
    else
        docker compose -f docker-compose.nonssl.gpu.yaml down
    fi

# GPU not detected - tear down CPU only
else
    echo "🚫 No GPU detected or NVIDIA drivers missing. Tearing down CPU-only version...\n\n"

    # Tear down SSL or non-SSL version appropriately
    if [ "$USE_SSL" = "true" ]; then
        docker compose -f docker-compose.ssl.nongpu.yaml down
    else
        docker compose -f docker-compose.nonssl.nongpu.yaml down
    fi
fi
