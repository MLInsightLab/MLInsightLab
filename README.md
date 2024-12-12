# ML Insight Lab

**The data science platform built *by* data scientists *for* data scientists**

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

ML Insight Lab is designed to provide a scalable, integrated environment for data scientists and machine learning engineers to perform distributed computing, experiment tracking, model serving, and inference management. The platform is tailored to handle end-to-end workflows, from interactive development in JupyterHub to production-ready model serving.

This project is a comprehensive Docker Compose setup for ML Insight Lab. It integrates multiple services, including Dask, Jupyter, MLflow, a custom API Hub, and a web UI - all served behind Nginx as a reverse proxy, providing a robust environment for data science, machine learning, and model management.


## Capabilities

1. **JupyterHub**:
   - Interactive, multi-user JupyterLab environment.
   - Automatically integrated with the other services in this platform (particularly MLflow and Dask).

2. **MLflow**:
   - Experiment tracking and machine learning model registry.
   - Stores experiment artifacts and metadata for reproducibility and model management.
   - Enables models to be easily served to the API hub.
   - All users with the role of admin or data scientist have unilateral access to the MLflow instance, enabling easy collaboration amongst the entire team.

3. **Dask**:
   - Python-native distributed computing framework for parallel computing.
   - Includes a Dask scheduler and multiple Dask workers for efficient task distribution.

4. **API Hub**:
   - Serves as the centralized API service for the platform
   - Serves trained models for inference, without users having to configure infrastructure.
   - Supports multiple model flavors including `pyfunc`, `sklearn`, `transformers`, and `hfhub`.
   - Provides endpoints for model loading, prediction, and management.
   - Also provides authentication, user management, and server management endpoints.

5. **Web UI**:
   - Provides a simple user interface to access other resources and perform actions within the platform.

6. **Nginx**:
   - Acts as a reverse proxy to manage and secure HTTP/HTTPS requests.

7. **Data Store**:
   - The `/data` directory in the JupyterHub, Dask, and Model Server services are all shared, providing a location for data sharing.
   - Data files can be uploaded and downloaded from the `/data` directory using the API.

8. **Variable Store**:
   - Store secure variables and other values to the Variable Store.
   - Complete with API access.

9. **mlinsightlab Python SDK**
   - Python SDK for interaction with the platform
   - Enables users to perform actions within the platform using a Python-native approach.
   - Installable via PyPi via the command `pip install mlinsightlab` or directly via source code install via this repository.

## Quick Start Guide

### Prerequisites

- Docker and Docker Compose installed on your system.
- Nvidia drivers and appropriate Nvidia Docker runtime installed (optional - GPU only).

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/jacobrenn/mlinsightlab.git
    cd mlinsightlab
    ```

2. **Create a `.env` file**:

    Use the provided example `.env` file and adjust the environment variables as needed.

    *Note that the provided `.env` file is comprehensive and requires no additional setup, but does leave security vulnerabilities due to default passwords and API keys.*

    ```bash
    cp env.example .env
    ```

3. **(Optional) Configure SSL Certificates**
   If you would like the Lab to be deployed using SSL termination, you will need to have your certificate `.pem` files saved to the directory `/{path/to/mlinsightlab}/certs`

   This can be accomplished by physically saving the files to this directory, or by having the files saved via a symbolic link.

4. **Pull and start the services**:

    The following command will both pull all required containers and start the service:

    ```bash
    docker-compose -f {chosen-docker-compose-file} up -d
    ```

    *Note that there are four options for which docker compose file can be used. Each one is appropriately named respective of whether SSL and GPU are used.*

### Stopping the Services

To stop and remove the services, run:

```bash
docker-compose -f {chosen-docker-compose-file} down
```

## Configuration

When deploying the Lab to a production (i.e. non-testing) environment, it is recommended that the following default environment variables be changed:

- MODEL_SERVER_ADMIN_USERNAME
   - This is the default username for the initial admin for the platform.
   - Default Value: `admin`
- MODEL_SERVER_ADMIN_PASSWORD
   - This is the default password for the initial admin for the platform. It is recommended to alter this to a preferred, secure value either when the platform is stood up initially or after the platform is created.
   - Default Value: `password`
- MODEL_SERVER_ADMIN_KEY
   - This is the default API key for the initial admin for the platform. It is recommended to alter this to a preferred, secure value either when the platform is stood up initially or after the platform is created. To change the value after the platform has been stood up, you will need to call the API directly or use the Python client to do so to issue a new API key.
   - Default Value: `mlil-admin-key`

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
