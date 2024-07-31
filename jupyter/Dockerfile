FROM python:latest

# Update software
RUN apt update && apt upgrade -y && apt autoremove -y
RUN apt install git emacs htop tmux sudo cron -y
RUN apt install -y ca-certificates curl gnupg
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
RUN apt update
RUN apt install nodejs -y

# Install configurable-http-proxy
RUN npm install -g configurable-http-proxy

# Copy and install requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Install jupyterlab_iframe extension
RUN pip install --no-cache-dir jupyterlab_iframe

# Install the variable inspector extension
RUN pip install --no-cache-dir lckr_jupyterlab_variableinspector

# Install the jupyterlab-git extension
RUN pip install --no-cache-dir jupyterlab-git

# Install the jupyterlab system monitor extension
RUN pip install --no-cache-dir jupyterlab-system-monitor
RUN jupyter labextension install jupyterlab-system-monitor

# Install the cron scheduler extension
RUN pip install --no-cache-dir jupyterlab_scheduler
RUN jupyter labextension install jupyterlab_scheduler

# Add sudo users to not require password for sudo
RUN echo "%sudo ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/root

# Expose the port jupyter runs on
EXPOSE 8000

# Make the /srv/jupyter directory and move the config to that directory
COPY jupyterhub_config.py /srv/jupyter/jupyterhub_config.py

# Copy the templates directory to the container's file system
COPY templates /srv/jupyter/templates

# Move the jupyterhub-singleuser.sh script to the /srv/jupyter directory and ensure it's executable
COPY jupyterhub-singleuser.sh /srv/jupyter/jupyterhub-singleuser.sh
RUN chmod +x /srv/jupyter/jupyterhub-singleuser.sh

# Copy the tutorials
COPY tutorials/ /notebooks

# Copy the start script and ensure it's executable
COPY start_jupyter.sh /code/start_jupyter.sh
RUN chmod +x /code/start_jupyter.sh

# Set the working directory
WORKDIR /home

CMD [ "sh", "/code/start_jupyter.sh" ]
