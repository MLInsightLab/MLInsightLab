FROM quay.io/jupyterhub/jupyterhub:latest

# Update software
RUN apt update && apt upgrade -y && apt autoremove -y
RUN apt install git emacs htop sudo cron -y

# Copy and install requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade -r /tmp/requirements.txt

# Add sudo users to not require password for sudo
RUN echo "%sudo ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/root

# Expose the port jupyter runs on
EXPOSE 8000

# Make the /root/.jupyter directory and move the config to that directory
COPY jupyterhub_config.py /srv/jupyter/jupyterhub_config.py

# Copy the start script and ensure it's executable
COPY start_jupyter.sh /code/start_jupyter.sh
RUN chmod +x /code/start_jupyter.sh

# Set the working directory
WORKDIR /home

CMD [ "sh", "/code/start_jupyter.sh" ]
