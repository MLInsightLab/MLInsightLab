#!/bin/bash

# Add environment variables
echo "export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI" > /etc/profile.d/mlflow.sh
echo "export DASK_SCHEDULER_ADDRESS=$DASK_SCHEDULER_ADDRESS" > /etc/profile.d/dask.sh
echo "export API_URL=$API_URL" > /etc/profile.d/api.sh

# Add read and write permissions to /notebooks and /data
chmod 1777 /notebooks
chmod 1777 /data

# Start the cron service
service cron restart

# Start JupyterHub
jupyterhub -f /srv/jupyter/jupyterhub_config.py --ip 0.0.0.0
