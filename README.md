# Open Data Science Platform (ODSP)

This project is a comprehensive Docker Compose setup for an Open Data Science Platform (ODSP). It integrates multiple services, including Dask, Jupyter, MLFlow, a model server, and Nginx, to provide a robust environment for data science, machine learning, and model management.

## Capabilities

1. **Dask**:
   - Distributed computing framework for parallel computing.
   - Includes a Dask scheduler and multiple Dask workers for efficient task distribution.

2. **Jupyter**:
   - Interactive notebooks for data analysis and visualization.
   - Integrated with Dask for parallel computing capabilities.

3. **MLFlow**:
   - Experiment tracking, model registry, and model serving.
   - Stores experiment artifacts and metadata for reproducibility and model management.

4. **Model Server**:
   - Serves trained models using MLFlow for inference.
   - Supports multiple model flavors including `pyfunc` and `sklearn`.
   - Provides endpoints for model loading, prediction, and management.

5. **Nginx**:
   - Acts as a reverse proxy to manage and secure HTTP requests.
   - Provides basic authentication for the Jupyter and MLFlow interfaces.

## Quick Start Guide

### Prerequisites

- Docker and Docker Compose installed on your system.

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-repo/odsp.git
    cd odsp
    ```

2. **Create a `.env` file**:

    Use the provided example `.env` file and adjust the environment variables as needed.

    ```bash
    cp .env.example .env
    ```

3. **Build and start the services**:

    ```bash
    docker-compose up -d
    ```

4. **Access the services**:

    - **Jupyter Notebook**: Open your web browser and go to `http://localhost:80`. Use the username and password set in the `.env` file for authentication.
    - **MLFlow Tracking UI**: Accessible at `http://localhost:2244`.

### Services Configuration

- **Dask Scheduler**: The main Dask scheduler responsible for task scheduling.
- **Dask Worker**: Workers that execute tasks assigned by the scheduler. The number of replicas can be configured in the `.env` file.
- **Jupyter**: Provides an interactive development environment with Dask integration.
- **MLFlow**: Tracks experiments and manages the model registry.
- **Model Server**: Serves trained models for inference.
- **Nginx**: Acts as a reverse proxy with basic authentication.

### Environment Variables

Ensure the following environment variables are correctly set in your `.env` file:

- **Dask**:
  - `DASK_IMAGE`: Docker image for Dask.
  - `DASK_SCHEDULER_ADDRESS`: Address of the Dask scheduler.
  - `DASK_WORKER_REPLICAS`: Number of Dask worker replicas.

- **Jupyter**:
  - `JUPYTER_IMAGE`: Docker image for Jupyter.
  - `JUPYTER_USERNAME`: Username for Jupyter authentication.

- **MLFlow**:
  - `MLFLOW_IMAGE`: Docker image for MLFlow.
  - `MLFLOW_BACKEND_STORE_URI`: URI for the MLFlow backend store.
  - `MLFLOW_TRACKING_ARTIFACT_STORE`: URI for the MLFlow artifact store.
  - `MLFLOW_TRACKING_URI`: URI for the MLFlow tracking server.

- **Model Server**:
  - `MODEL_SERVER_IMAGE`: Docker image for the model server.

- **Nginx**:
  - `NGINX_IMAGE`: Docker image for Nginx.
  - `NGINX_USERNAME`: Username for Nginx authentication.
  - `NGINX_PASSWORD`: Password for Nginx authentication.

- **Volume Mounts**:
  - `MLFLOW_ARTIFACT_STORAGE`: Volume for MLFlow artifact storage.
  - `MLFLOW_BACKEND_STORAGE`: Volume for MLFlow backend storage.
  - `NOTEBOOK_MOUNT`: Volume for notebook storage.

### Stopping the Services

To stop and remove the services, run:

```bash
docker-compose down
```

### Scaling Dask Workers

To scale the number of Dask workers, adjust the `DASK_WORKER_REPLICAS` in your `.env` file and re-deploy:

```bash
docker-compose up -d --scale dask-worker=<number_of_workers>
```

## Model Server Details

The model server is built using FastAPI and is designed to serve machine learning models registered with MLFlow. The server provides several endpoints for managing and serving models.

### Endpoints

1. **Redirect to API Documentation**:

    Redirects the root URL to the API documentation.

    ```http
    GET /
    ```

2. **Load a Model**:

    Loads a model from the MLFlow server.

    ```http
    GET /load_model/{model_name}/{model_flavor}/{model_version_or_alias}
    ```

3. **List Loaded Models**:

    Lists all currently loaded models.

    ```http
    GET /loaded_models
    ```

4. **Unload a Model**:

    Unloads a model from memory.

    ```http
    DELETE /{model_name}/{model_flavor}/{model_version_or_alias}
    ```

5. **Predict using a Model**:

    Performs prediction using a specified model version or alias.

    ```http
    POST /{model_name}/{model_version_or_alias}
    ```

### Request and Response Formats

#### PredictRequest

The `PredictRequest` model is used for prediction requests.

```python
class PredictRequest(BaseModel):
    data: list
    predict_function: str = 'predict'
    model_flavor: str = 'pyfunc'
    dtype: str = None
    params: dict = None
```

- **data**: List of input data for prediction.
- **predict_function**: Prediction function to use (`predict` or `predict_proba`).
- **model_flavor**: Model flavor (`pyfunc` or `sklearn`).
- **dtype**: Data type for the input data.
- **params**: Additional parameters for the prediction function.

### Example Usage

1. **Load a Model**:

    ```http
    GET /load_model/my_model/pyfunc/1
    ```

2. **List Loaded Models**:

    ```http
    GET /loaded_models
    ```

3. **Predict with a Loaded Model**:

    ```http
    POST /my_model/1
    Content-Type: application/json

    {
        "data": [[1, 2, 3], [4, 5, 6]],
        "predict_function": "predict",
        "model_flavor": "pyfunc"
    }
    ```

### Notes

- This setup assumes that you have Docker and Docker Compose installed on your system.
- Ensure that all necessary environment variables are set in the `.env` file before deploying the services.
- The model server automatically installs required dependencies for models based on the `requirements.txt` file stored with the model in MLFlow.
