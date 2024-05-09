# Open Data Science Platform

The Open Data Science Platform is a Docker-compose based project that aims to provide a scalable and integrated environment for data science tasks including distributed computing, machine learning experimentation, model deployment, and more.

## Components

The platform consists of the following services:

- **Dask Scheduler and Workers**: Provides distributed computing capabilities using Dask, allowing parallel computation on large datasets.
- **Jupyter Notebook Server**: Offers an interactive environment for data exploration, analysis, and visualization, with integration with Dask for distributed computing.
- **MLflow Tracking Server**: Enables experiment tracking and management, allowing data scientists to log parameters, metrics, and artifacts during model training.
- **Model Server**: Hosts machine learning models for inference, providing a RESTful API for predictions.

## Installation

To set up the Open Data Science Platform on your local machine, follow these steps:

1. Clone this repository:

```bash
git clone https://github.com/jacobrenn/OpenDataSciencePlatform
```

2. Navigate to the project directory:

```bash
cd OpenDataSciencePlatform
```


3. Make sure you have Docker and Docker Compose installed on your machine.

4. Run the following command to start the services:

```bash
docker compose up -d
```

## Accessing Services

Once the services are up and running, you can access them as follows:

- **Jupyter Notebook Server**: Open your web browser and go to `http://localhost`.
- **MLflow Tracking Server**: Access MLflow UI by navigating to `http://localhost/mlflow` in your web browser.
- **Model Server**: Once MLflow Tracking Server is up and running, models deployed using MLflow can be accessed through the Model Server. Example endpoint: `http://localhost/inference`.

## Additional Notes

- Make sure to shut down the services when not in use to conserve resources
