FROM python:latest

# Update software
RUN apt update && apt upgrade -y && apt autoremove -y
RUN apt install git emacs htop sudo cron nodejs npm -y

# Install configurable-http-proxy
RUN npm install -g configurable-http-proxy

# Copy and install requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Add sudo users to not require password for sudo
RUN echo "%sudo ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/root

# Expose the port jupyter runs on
EXPOSE 8000

# Make the /srv/jupyter directory and move the config to that directory
COPY jupyterhub_config.py /srv/jupyter/jupyterhub_config.py

# Move the jupyterhub-singleuser.sh script to the /srv/jupyter directory and ensure it's executable
COPY jupyterhub-singleuser.sh /srv/jupyter/jupyterhub-singleuser.sh
RUN chmod +x /srv/jupyter/jupyterhub-singleuser.sh

# Copy the start script and ensure it's executable
COPY start_jupyter.sh /code/start_jupyter.sh
RUN chmod +x /code/start_jupyter.sh

# Set the working directory
WORKDIR /home

CMD [ "sh", "/code/start_jupyter.sh" ]
