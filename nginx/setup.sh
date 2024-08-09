#!/bin/bash

# Check for an environment variable that determines SSL mode
if [ "$USE_SSL" == "true" ]; then
    echo "Starting Nginx with SSL configuration"
    # Copy or link the SSL configuration file to the default location
    cp /etc/nginx/nginx.ssl.conf /etc/nginx/nginx.conf
else
    echo "Starting Nginx with Non-SSL configuration"
    # Copy or link the Non-SSL configuration file to the default location
    cp /etc/nginx/nginx.nonssl.conf /etc/nginx/nginx.conf
fi

echo "Starting Nginx"
nginx -g 'daemon off;'
