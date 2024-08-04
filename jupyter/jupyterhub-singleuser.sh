#!/bin/bash -l

source /etc/profile.d/dask.sh
source /etc/profile.d/mlflow.sh
source /etc/profile.d/api.sh

exec jupyterhub-singleuser
