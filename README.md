# ML Insight Lab

**The data science platform built *by* data scientists *for* data scientists**

[![Join Slack](https://img.shields.io/badge/slack-join-blue?logo=slack)](https://join.slack.com/t/mlinsightlab/shared_invite/zt-35ovs382b-tOR1MY6c2ExhHzkyJeVWhQ)
![License](https://img.shields.io/github/license/mlinsightlab/mlinsightlab)
![PyPI](https://img.shields.io/pypi/v/mlinsightlab)
![GitHub stars](https://img.shields.io/github/stars/mlinsightlab/mlinsightlab?style=social)

## Service Status

| Service       | Status Badge |
|---------------|--------------|
| JupyterHub    | [![JupyterHub CI](https://github.com/mlinsightlab/mlinsightlab-jupyter/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/mlinsightlab/mlinsightlab-jupyter/actions/workflows/docker-publish.yml) |
| MLflow        | [![MLflow CI](https://github.com/mlinsightlab/mlinsightlab-mlflow/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/mlinsightlab/mlinsightlab-mlflow/actions/workflows/docker-publish.yml) |
| Dask          | [![Dask CI](https://github.com/mlinsightlab/mlinsightlab-dask/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/mlinsightlab/mlinsightlab-dask/actions/workflows/docker-publish.yml) |
| API Hub       | [![API Hub CI](https://github.com/mlinsightlab/mlinsightlab-apihub/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/mlinsightlab/mlinsightlab-apihub/actions/workflows/docker-publish.yml) |
| Web UI        | [![Web UI CI](https://github.com/mlinsightlab/mlinsightlab-ui/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/mlinsightlab/mlinsightlab-ui/actions/workflows/docker-publish.yml) |
| Nginx         | [![NGINX CI](https://github.com/mlinsightlab/mlinsightlab-nginx/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/mlinsightlab/mlinsightlab-nginx/actions/workflows/docker-publish.yml) |


## Table of Contents

- [Overview](#overview)
- [Capabilities](#capabilities)
- [Quick Start Guide](#quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Stopping the Services](#stopping-the-services)
- [Configuration](#configuration)
- [Logs and Debugging](#logs-and-debugging)
- [Security Considerations](#security-considerations)
- [Contributing](#contributing)
- [License](#license)

## Overview

ML Insight Lab provides a scalable, containerized environment for data scientists and machine learning engineers. It supports distributed computing, experiment tracking, model serving, and inference workflows from development to production—all behind a secure Nginx reverse proxy.

ML Insight Lab is designed in a modular way so that certain services can be deployed or not at the administrator's discretion.

The platform includes:

- Interactive development (JupyterHub)
   - Integrated with the API Hub for user authentication and authorization.
- Experiment tracking (MLflow)
- Distributed computing (Dask)
- Model serving and user management (API Hub)
   - Serves as a centralized service for user authentication and authorization, as well as a centralized service to deploy models to the platform.
- Artifact and file storage (MinIO)
   - Two deployments of MinIO are provided. The first one is accessible only to the internal services in the platform. An external service is also accessible to serve as artifact storage for users of the platform, and the platform's API Hub manages access to this service.
- Independent Postgres backends per service for clean data separation
   - These deployments are only accessible to the internal services in the platform

## Capabilities

### 1. **JupyterHub**
- Multi-user JupyterLab environment with automatic integration to Dask and MLflow
- Uses its own dedicated PostgreSQL backend

### 2. **MLflow**
- Full experiment tracking and model registry
- Dedicated PostgreSQL backend for metadata
- Stores artifacts (e.g., models, logs) in **MinIO S3-compatible storage**
- Models can be served directly to the API Hub
- Accessible to all team members with "admin" or "data scientist" roles

### 3. **Dask**
- Python-native distributed task scheduler
- Automatically scaled with scheduler and worker containers
- Integrated with Jupyter for scalable notebook execution

### 4. **API Hub**
- Central API service to expose models for inference
- Supports multiple model types: pyfunc, sklearn, transformers, hfhub
- Provides endpoints for prediction, model loading, user authentication, and system management
- Uses:
  - A dedicated PostgreSQL instance
  - MinIO for user-uploaded files and model storage

### 5. **Web UI**
- Unified interface for accessing services and managing tasks

### 6. **Nginx**
- Reverse proxy handling HTTP and HTTPS traffic
- Automatically detects and routes based on SSL certificate availability

### 7. **Data Store**
- An externally-accessible MinIO instance is deployed to serve as file storage within the platform.
- This instance is managed by the platform's API Hub to create, delete, and otherwise manage users.
- Accessible via API for file upload/download

### 8. **Variable Store**
- Central key-value store accessible via API for storing environment-specific variables

### 9. **mlinsightlab Python SDK**
- Lightweight Python SDK preinstalled in JupyterHub.
- Allows Python-native interaction with the API Hub and other platform features.
- Documentation for the SDK can be found at [this site](https://mlinsightlab.github.io/MLInsightLab-Python-SDK/)
- To see examples of how to use the SDK with the platform, please see [this repository](https://github.com/mlinsightlab/mlinsightlab-examples)
- Installable via:

   ```bash
   pip install mlinsightlab
   ```


## Quick Start Guide

### Prerequisites

- Docker and Docker Compose installed on your system.
- Nvidia drivers and appropriate Nvidia Docker runtime installed (optional - GPU only).

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/mlinsightlab/mlinsightlab.git
    cd mlinsightlab
    ```

2. **Create a `.env` file**:

    Use the provided example `env.example` file and adjust the environment variables as needed.

    *Note that the provided `env.example` file is comprehensive and requires no additional setup, but does leave security vulnerabilities due to default passwords and API keys.*

    ```bash
    cp env.example .env
    ```

    By its default configuration in the provided `env.example` file, the platform is designed to be deployed locally. If you are deploying to a cloud server, you will need to alter the `HOST` variable in the `env.example` file to ensure that the service knows it is being deployed to another host.

    Additionally, if you are deploying the externally-facing MinIO services to the platform, the API is deployed at `s3.{HOST}`. You will need to ensure that you have the correct DNS records set up to ensure the service is accessible. By default, the Lab's API Hub handles user management and authentication for the storage service, and user credentials for the storage service match their credentials for the Lab. If you would like to change that and manage the service yourself, set `API_HUB_MANAGE_STORAGE=false` in your `.env` file.

3. **(Optional) Configure SSL Certificates**
   If you would like the Lab to be deployed using SSL termination, you will need to have your certificate `.pem` files saved to the directory `/{path/to/mlinsightlab}/certs`

   This can be accomplished by physically saving the files to this directory, or by having the files saved via a symbolic link.

   For most deployments, we find using [Lets Encrypt](https://letsencrypt.org) effective.

4. **Pull and start the services**:

   For ease of deployment and teardown, we have provided two shell scripts - `deploy.sh` and `teardown.sh`, respectively. These scripts will identify whether TLS certificates have been installed and whether GPU resources are available on your machine and automatically deploy the correct configuration of the platform. To deploy the platform, simply run the following:

   ```bash
   sh deploy.sh
   ```

   If you would like to deploy the platform using docker compose directly, then you will have to identify whether GPU and SSL support are to be used. The platform is deployed via a combination of three docker compose files. Using your chosen compose files, please modify the following command:

   ```bash
   docker compose -f docker-compose.base.yaml -f {chosen_ssl_or_nonssl_compose_file} -f {chosen_gpu_or_nongpu_compose_file} up -d
   ```

### Stopping the Services

   To stop and remove the services using the teardown script provided, run:

   ```bash
   sh teardown.sh
   ```

   Or, if you would like to teardown the platform using docker compose directly, then run the following:

   ```bash
   docker-compose -f docker-compose.base.yaml -f {chosen_ssl_or_nonssl_compose_file} -f {chosen_gpu_or_nongpu_compose_file} down
   ```

## Configuration

The platform relies on environment variables defined in the `.env` file. Below is a list of important variables that **should be updated before deploying to production**, especially those related to authentication and database access. Note that we provide the file `env.example` as a starting point.

### Authentication and Admin Credentials

| Variable                  | Default Value        | Description                                      |
|--------------------------|----------------------|--------------------------------------------------|
| `API_HUB_ADMIN_USERNAME` | `admin`              | Default admin username for API Hub              |
| `API_HUB_ADMIN_PASSWORD` | `password`           | Default admin password for API Hub              |

---

### JupyterHub Configuration

| Variable                      | Default Value         | Description                                         |
|------------------------------|-----------------------|-----------------------------------------------------|
| `JUPYTER_POSTGRES_USER`      | `jupyterhub`          | Postgres username for JupyterHub                   |
| `JUPYTER_POSTGRES_PASSWORD`  | `jupyterhub`          | Postgres password for JupyterHub                   |
| `JUPYTER_POSTGRES_DB`        | `jupyterhub`          | Postgres database name for JupyterHub              |

---

### MLflow Configuration

| Variable                           | Default Value                                   | Description                                      |
|-----------------------------------|-------------------------------------------------|--------------------------------------------------|
| `MLFLOW_BACKEND_STORE_URI`        | `postgresql://${MLFLOW_POSTGRES_USER}:${MLFLOW_POSTGRES_PASSWORD}@postgres-mlflow:5432/${MLFLOW_POSTGRES_DB}` | MLflow's Postgres URI for metadata |
| `MLFLOW_TRACKING_ARTIFACT_STORE`  | `s3://mlflow`                                   | MinIO bucket for MLflow artifacts               |

#### MLflow Postgres Credentials

| Variable                     | Default Value     | Description                            |
|-----------------------------|-------------------|----------------------------------------|
| `MLFLOW_POSTGRES_USER`      | `mlflow_postgres` | Username for MLflow's Postgres backend |
| `MLFLOW_POSTGRES_PASSWORD`  | `mlflow_postgres` | Password for MLflow's Postgres backend |
| `MLFLOW_POSTGRES_DB`        | `mlflow_postgres` | Database name for MLflow               |

---

### API Hub Configuration

| Variable                        | Default Value           | Description                                 |
|--------------------------------|-------------------------|---------------------------------------------|
| `API_HUB_POSTGRES_USER`        | `apihub_postgres`       | Username for API Hub's Postgres             |
| `API_HUB_POSTGRES_PASSWORD`    | `apihub_postgres`       | Password for API Hub's Postgres             |
| `API_HUB_POSTGRES_DB`          | `apihub_postgres`       | Database name for API Hub                   |

---

### Internal MinIO Configuration

| Variable               | Default Value      | Description                                      |
|------------------------|--------------------|--------------------------------------------------|
| `MINIO_ROOT_USER_INTERNAL`      | `minioadmin`       | MinIO root username (S3 access key)              |
| `MINIO_ROOT_PASSWORD_EXTERNAL`  | `minioadmin`       | MinIO root password (S3 secret key)              |


---

### External MinIO Configuration

| Variable               | Default Value      | Description                                      |
|------------------------|--------------------|--------------------------------------------------|
| `MINIO_ROOT_USER_EXTERNAL`      | `minioadmin`       | MinIO root username (S3 access key)              |
| `MINIO_ROOT_PASSWORD_EXTERNAL`  | `minioadmin`       | MinIO root password (S3 secret key)              |


---

## Logs and Debugging

   To view logs for any service, use the following command:

   ```bash
   docker-compose -f {chosen-docker-compose-file} logs {service-name}
   ```

   For example, to view the logs for the JupyterHub service:

   ```bash
   docker-compose -f {chosen-docker-compose-file} logs jupyter
   ```

## Security Considerations

   1. **Environment Variables**: Avoid using default passwords and API keys in production. Update them in the `.env` file.
   2. **SSL/HTTPS**: Ensure SSL certificates are properly configured for secure connections.
   3. **User Management**: If deploying JupyterHub for multiple users, ensure appropriate permissions and roles are assigned.

## Contributing

   We welcome contributions! If you'd like to contribute to this project, please follow these steps:

   1. Fork the repository.
   2. Create a new feature branch (`git checkout -b feature/your-feature`).
   3. Commit your changes (`git commit -m 'Add your feature'`).
   4. Push the branch (`git push origin feature/your-feature`).
   5. Open a pull request.

   Please ensure your code follows our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project includes third-party components:

- **PostgreSQL** - [PostgreSQL License](https://www.postgresql.org/about/licence/)
- **MinIO** - [GNU Affero General Public License v3.0 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.html)
- **MinIO Command Line Client (mc)** - [GNU Affero General Publice Licene v3.0 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.html)

MinIO and mc are used "out-of-the-box" with no alterations or adaptations by us. Furthermore, the platform can be easily configured to utilize other S3-compatible storage mechanisms. If you redistribute the platform with modifications to MinIO, please review your obligations under the AGPLv3.
