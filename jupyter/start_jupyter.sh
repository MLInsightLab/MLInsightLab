#!/bin/bash

# Add environment variables
echo "export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI" >> /etc/profile.d/mlflow.sh
echo "export DASK_SCHEDULER_ADDRESS=$DASK_SCHEDULER_ADDRESS" >> /etc/profile.d/dask.sh

# Assign read and write rights to all folders within the /home directory
chmod -R +r /home
chmod -R +w /home

# Start JupyterHub
jupyterhub -f /srv/jupyter/jupyterhub_config.py --ip 0.0.0.0
