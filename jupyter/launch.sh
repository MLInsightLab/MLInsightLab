#!/bin/bash

# Check if the environment variable JUPYTER_ENV exists
if [ -z "$JUPYTER_ENV" ]; then
  echo "JUPYTER_ENV is not set"
  exit 1
fi

# create odsp user
adduser --disabled-password --gecos "" odsp
adduser odsp sudo

# Allow root and odsp users to run sudo commands without password
echo "root ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/root
echo "odsp ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/odsp

# Add MLFLOW_TRACKING_URI and DASK_SCHEDULER_ADDRESS to odsp user
echo "export MLFLOW_TRACKING_URI=http://mlflow:2244" >> /home/odsp/.bashrc
echo "export DASK_SCHEDULER_ADDRESS=http://dask-scheduler:8786" >> /home/odsp/.bashrc

# Check if the value of JUPYTER_ENV is "jupyterlab"
if [ "$JUPYTER_ENV" = "jupyterlab" ]; then
    echo "JUPYTER_ENV is set to jupyterlab"

    # Run JupyterLab as odsp user
    sudo -u odsp bash -c "source /home/odsdp/.bashrc && jupyter-lab --ip 0.0.0.0 --port 8000 --allow-root --notebook-dir=/home --ServerApp.token='' --ServerApp.password=''"

# Check if the value of JUPYTER_ENV is "jupyterhub"
elif [ "$JUPYTER_ENV" = "jupyterhub" ]; then
    echo "JUPYTER_ENV is set to jupyterhub"

    # Run JupyterHub as root user
    jupyterhub -f /srv/jupyterhub/jupyterhub_config.py --ip 0.0.0.0

# Else, the value is not understood
else
    echo "JUPYTER_ENV set to unrecognized value: $JUPYTER_ENV"
    exit 1
fi
