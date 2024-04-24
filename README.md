# Open Data Science Platform
The Open Data Science Platform is a comprehensive setup for a scalable data science and machine learning environment using Docker Compose. It includes components for distributed computing, Jupyter notebooks, MLflow tracking and model management, a model serving API, and a web application.

# Components
1. Dask Scheduler and Workers

Dask is used for parallel computing. The Dask scheduler coordinates the execution of tasks, while workers perform the computations. The number of workers can be scaled according to workload.

- dask-scheduler: Runs the Dask scheduler.
- dask-worker: Dask workers that connect to the scheduler.

2. Jupyter Notebook Server

Jupyter provides an interactive computing environment for data science tasks. It is accessible via a web browser and allows users to create and share documents containing live code, equations, visualizations, and narrative text.

- jupyter: Jupyter Notebook server.

3. MLflow Tracking Server

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. The tracking server logs and tracks experiments, parameters, and metrics.

- mlflow: MLflow tracking server.

4. Model Serving API

A simple API for serving machine learning models.

- model-server: Model serving API.

5. Web Application

A web application that interacts with the deployed models.

- webapp: Web application.

# Setup Instructions
## Prerequisites
- Docker Engine
- Docker Compose

## Installation
1. Clone this repository:
```bash
git clone https://github.com/jacobrenn/OpenDataSciencePlatform.git
cd OpenDataSciencePlatform
```

2. Run Docker Compose:
```bash
docker compose build
docker compose up -d
```

3. Access services:
- Jupyter Notebook: http://localhost:8000
- MLflow Tracking: http://localhost:2244
- Model Server API: http://localhost:4488
- Web Application: http://localhost

## Additional Configuration
- GPU Support: Uncomment the GPU-related lines in the docker-compose.yaml file to enable GPU support.
- Volume Configuration: Volume mounts are used for persistence. Customize volume paths as needed in the docker-compose.yaml file.

# Notes
Make sure ports 8000, 2244, 4488, and 80 are available and not in use by other services on your system.
For detailed configuration options and advanced usage, refer to the official documentation of each component.