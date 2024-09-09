# MLInsightLab

## Table of Contents

- [Overview](#overview)
- [Capabilities](#capabilities)
- [Quick Start Guide](#quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Stopping the Services](#stopping-the-services)
- [Configuration](#configuration)
- [Logs and Debugging](#logs-and-debugging)
- [Usage Examples](#usage-examples)
- [Security Considerations](#security-considerations)
- [Contributing](#contributing)
- [License](#license)

## Overview

MLInsightLab is designed to provide a scalable, integrated environment for data scientists and machine learning engineers to perform distributed computing, experiment tracking, model serving, and inference management. The platform is tailored to handle end-to-end workflows, from interactive development in JupyterHub to production-ready model serving.

This project is a comprehensive Docker Compose setup for MLInsightLab. It integrates multiple services, including Dask, Jupyter, MLFlow, a custom model server and web UI, and Nginx, providing a robust environment for data science, machine learning, and model management.


## Capabilities

1. **JupyterHub**:
   - Interactive, multi-user JupyterLab environment.
   - Automatically integrated with the other services in this platform.

2. **MLFlow**:
   - Experiment tracking and machine learning model registry.
   - Stores experiment artifacts and metadata for reproducibility and model management.

3. **Dask**:
   - Python-native distributed computing framework for parallel computing.
   - Includes a Dask scheduler and multiple Dask workers for efficient task distribution.

4. **Model Server**:
   - Serves trained models using MLFlow for inference, without users having to configure infrastructure.
   - Supports multiple model flavors including `pyfunc`, `sklearn`, `transformers`, and `hfhub`.
   - Provides endpoints for model loading, prediction, and management.
   - Also provides authentication, user management, and server management endpoints.

5. **Web UI**:
   - Provides a simple user interface to access other resources

6. **Nginx**:
   - Acts as a reverse proxy to manage and secure HTTP/HTTPS requests.

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
    cp .env.example .env
    ```

3. **(Optional) Configure SSL Certificates**
   If you would like the Lab to be deployed using SSL termination, you will need to have your certificate `.pem` files saved to the directory `{path-to-mlinsightlab-directory}/certs`

   This can be accomplished by physically saving the files to this directory, or by having the files saved via a symbolic link.

4. **Build and start the services**:

    The following command will both build all required containers and start the service:

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
   - This is the default username for the initial admin for the platform. It is recommended to alter this to a preferred value when the platform is stood up.
   - Default Value: `admin`
- MODEL_SERVER_ADMIN_PASSWORD
   - This is the default password for the initial admin for the platform. It is recommended to alter this to a preferred, secure value either when the platform is stood up initially or after the platform is created.
   - Default Value: `password`
- MODEL_SERVER_ADMIN_KEY
   - This is the default API key for the initial admin for the platform. It is recommended to alter this to a preferred, secure value either when the platform is stood up initially or after the platform is created.
   - Default Value: `mlil-admin-key`
- MODEL_SERVER_SYSTEM_KEY
   - This is the default API key used by the platform itself to allow services to communicate between one another. It is recommended that this be altered to a secure value when the platform is stood up initially. **NOTE THAT THIS VALUE CANNOT BE CHANGED AFTER THE PLATFORM IS STOOD UP**.
   - Default Value: `mlil-system-key`
- UI_SECRET_KEY
   - This secret key is used by the platform web UI to help secure communications. It is recommended that this value be altered to a secure value when the platform is stood up initially. **NOTE THAT THIS VALUE CANNOT BE CHANGED AFTER THE PLATFORM IS STOOD UP**.
   - Default Value: `mlil-ui`

## Logs and Debugging

To view logs for any service, use the following command:

```bash
docker-compose -f {chosen-docker-compose-file} logs {service-name}
```

For example, to view the logs for the JupyterHub service:

```bash
docker-compose -f {chosen-docker-compose-file} logs juptyer
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
