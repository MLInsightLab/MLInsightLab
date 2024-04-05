# Use Anaconda as the base image
FROM quay.io/jupyterhub/jupyterhub:latest

# Update software
RUN apt update && apt upgrade -y && apt autoremove -y
RUN apt install git emacs htop default-jre default-jdk -y

# Copy and install requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade -r /tmp/requirements.txt

# Add user "admin"
RUN adduser --disabled-password --gecos "" admin

# Set the working directory
WORKDIR /srv/jupyterhub

# Copy your configuration files to the container
# If you have a custom configuration file, you can copy it to the appropriate location
COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py

# Expose the port jupyterhub runs on
EXPOSE 8000

# Set the command to run JupyterHub
CMD ["jupyterhub", "-f", "/srv/jupyterhub/jupyterhub_config.py", "--ip", "0.0.0.0"]
