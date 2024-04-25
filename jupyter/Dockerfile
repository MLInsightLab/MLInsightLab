FROM quay.io/jupyterhub/jupyterhub:latest

# Update software
RUN apt update && apt upgrade -y && apt autoremove -y
RUN apt install git emacs htop sudo -y

# For spark
#RUN apt install python3.11 git emacs htop default-jre default-jdk -y
#RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Add Docker's official GPG key:
RUN apt-get update
RUN apt-get install ca-certificates curl
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
RUN chmod a+r /etc/apt/keyrings/docker.asc
    
# Add the repository to Apt sources:
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt update
RUN apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

# Copy and install requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade -r /tmp/requirements.txt

# Create odsp user
RUN adduser --disabled-password --gecos "" odsp
RUN adduser odsp sudo

# Add odsp and root users to not require password for sudo
RUN echo "root ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/root
RUN echo "odsp ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/odsp

# Add environment variables
RUN echo "export MLFLOW_TRACKING_URI=http://mlflow:2244" >> /home/odsp/.bashrc
RUN echo "export DASK_SCHEDULER_ADDRESS=tcp://dask-scheduler:8786" >> /home/odsp/.bashrc

# Expose the port jupyter runs on
EXPOSE 8000

USER odsp
WORKDIR /home
CMD ["jupyter-lab", "--ip", "0.0.0.0", "--port", "8000", "--notebook-dir=/home", "--ServerApp.token=''", "--ServerApp.password=''", "--allow_origin='*'", "--allow_remote_access=True"]
