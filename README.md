# Open Data Science Platform (ODSP)

This project is a comprehensive Docker Compose setup for an Open Data Science Platform (ODSP). It integrates multiple services, including Dask, Jupyter, MLFlow, a model server, and Nginx, to provide a robust environment for data science, machine learning, and model management.

## Capabilities

1. **Dask**:
   - Distributed computing framework for parallel computing.
   - Includes a Dask scheduler and multiple Dask workers for efficient task distribution.

2. **JupyterHub**:
   - Interactive, multi-user JupyterLab environment
   - Automatically integrated with the other services in this platform

3. **MLFlow**:
   - Experiment tracking, model registry, and model serving.
   - Stores experiment artifacts and metadata for reproducibility and model management.

4. **Model Server**:
   - Serves trained models using MLFlow for inference, without users having to configure infrastructure.
   - Supports multiple model flavors including `pyfunc`, `sklearn`, and `transformers`.
   - Provides endpoints for model loading, prediction, and management.

5. **Nginx**:
   - Acts as a reverse proxy to manage and secure HTTP requests.

## Quick Start Guide

### Prerequisites

- Docker and Docker Compose installed on your system.

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/jacobrenn/opendatascienceplatform.git
    cd odsp
    ```

2. **Create a `.env` file**:

    Use the provided example `.env` file and adjust the environment variables as needed.

    *Note that the provided `.env` file is comprehensive and requires no additional setup, but does leave security vulnerabilities due to default passwords and API keys*

    ```bash
    cp .env.example .env
    ```

3. **Build and start the services**:

    The following command will both build all required containers and start the service

    ```bash
    docker-compose up -d
    ```

### Services Configuration

- **Dask Scheduler**: The main Dask scheduler responsible for task scheduling.
- **Dask Worker**: Workers that execute tasks assigned by the scheduler. The number of replicas can be configured in the `.env` file.
- **Jupyter**: Provides an interactive development environment with Dask integration.
- **MLFlow**: Tracks experiments and manages the model registry.
- **Model Server**: Serves trained models for inference.
- **ODSP UI**: Front-end interface for interacting with models and the platform.
- **Nginx**: Acts as a reverse proxy with basic authentication.

### Stopping the Services

To stop and remove the services, run:

```bash
docker-compose down