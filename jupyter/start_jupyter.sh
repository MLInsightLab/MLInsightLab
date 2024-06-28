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
echo "JUPYTER_PASSWORD=$JUPYTER_PASSWORD" >> /etc/environment

# Copy the jupyter lab configuration file over to the user's home directory
mkdir /home/$NB_USER/.jupyter
cp /code/jupyter_lab_config.py /home/$NB_USER/.jupyter/jupyter_lab_config.py

# Start Jupyter Lab under the NB_USER
/bin/su --login -c "jupyter lab --notebook-dir=/home --ip=0.0.0.0 --port 8000 --no-browser --ServerApp.token='' --allow-root" $NB_USER
