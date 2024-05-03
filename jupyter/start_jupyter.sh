#!/bin/bash

# Define default username
: ${NB_USER:=odsp}

# Create user if not exists
id -u $NB_USER &>/dev/null || useradd -m -s /bin/bash $NB_USER

# Add user to sudoers
echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Start Jupyter Lab
sudo -E -u $NB_USER jupyter lab --ip=0.0.0.0 --allow-root --port 8000 --notebook-dir /home
