#!/bin/bash

# Check if the .htpasswd file exists
#if [ ! -f /etc/nginx/.htpasswd ]; then
    # Generate .htpasswd file
    #htpasswd -b -c /etc/nginx/.htpasswd $AUTH_USERNAME $AUTH_PASSWORD
    #echo "Password successfully configured"
#fi

echo "Starting nginx"
nginx -g 'daemon off;'
