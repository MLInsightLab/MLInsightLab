#!/bin/bash

# Check if the environment variable JUPYTER_ENV exists
if [ -z "$JUPYTER_ENV" ]; then
  echo "JUPYTER_ENV is not set"
  exit 1
fi

# Check if the value of JUPYTER_ENV is "jupyterlab"
if [ "$JUPYTER_ENV" = "jupyterlab" ]; then
    echo "JUPYTER_ENV is set to jupyterlab"
    jupyter-lab --ip 0.0.0.0 --port 8000 --allow-root --notebook-dir=/home --ServerApp.token='' --ServerApp.password=''

# Check if the value of JUPYTER_ENV is "jupyterhub"
elif [ "$JUPYTER_ENV" = "jupyterhub" ]; then
    echo "JUPYTER_ENV is set to jupyterhub"
    adduser --disabled-password --gecos "" odsp
    adduser odsp sudo
    
    # Allow root and odsp users to run sudo commands without password
    RUN echo "root ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/root
    RUN echo "odsp ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/odsp
    
    jupyterhub -f /srv/jupyterhub/jupyterhub_config.py --ip 0.0.0.0

# Else, the value is not understood
else
    echo "JUPYTER_ENV set to unrecognized value: $JUPYTER_ENV"
    exit 1
fi
