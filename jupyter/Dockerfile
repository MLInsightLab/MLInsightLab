# FROM quay.io/jupyterhub/jupyterhub:latest
FROM python:latest

# Update software
RUN apt update && apt upgrade -y && apt autoremove -y
RUN apt install git emacs htop -y
#RUN apt install python3.11 git emacs htop default-jre default-jdk -y
#RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Add Docker's official GPG key:
#RUN apt-get update
#RUN apt-get install ca-certificates curl
#RUN install -m 0755 -d /etc/apt/keyrings
#RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
#RUN chmod a+r /etc/apt/keyrings/docker.asc
    
# Add the repository to Apt sources:
#RUN echo \
  #"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  #$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  #tee /etc/apt/sources.list.d/docker.list > /dev/null
#RUN apt update
#RUN apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

# Copy and install requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade -r /tmp/requirements.txt

# Add user "admin"
#RUN adduser --disabled-password --gecos "" admin

# Expose the port jupyterlab runs on
EXPOSE 8000

# When using jupyterhub
# COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py

# Set the command to run JupyterLab
# CMD ["jupyterhub", "-f", "/srv/jupyterhub/jupyterhub_config.py", "--ip", "0.0.0.0"]
CMD ["jupyter-lab", "--ip", "0.0.0.0", "--port", "8000", "--allow-root", "--notebook-dir=/home", "--NotebookApp.token=''", "--NotebookApp.password=''"]
