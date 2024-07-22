#!/bin/bash -l

source /etc/profile.d/dask.sh
source /etc/profile.d/mlflow.sh

exec jupyterhub-singleuser "@"
