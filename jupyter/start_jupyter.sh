#!/bin/bash

# Define default username
: ${NB_USER:=odsp}

# Create user if not exists
id -u $NB_USER &>/dev/null || useradd -m -s /bin/bash $NB_USER
adduser $NB_USER sudo

# Add user to sudoers
echo "$NB_USER ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$NB_USER

# Add environment variables
echo "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI" >> /etc/environment
echo "DASK_SCHEDULER_ADDRESS=$DASK_SCHEDULER_ADDRESS" >> /etc/environment

# Ensure password is correctly written and created
python /code/create_password.py
password=$(cat $PASSWORD_FILE)

# Start Jupyter Lab under the NB_USER
/bin/su --login -c "jupyter lab --notebook-dir=/home --ip=0.0.0.0 --port 8000 --no-browser --ServerApp.token='' --ServerApp.password=$password --allow-root" $NB_USER
