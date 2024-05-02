# Use the official nginx image as base
FROM nginx:latest

# Update
RUN apt update
RUN apt upgrade -y
RUN apt install apache2-utils -y

# Copy the custom nginx configuration file into the container
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the password setup file to the container
COPY setup.sh /code/setup.sh

EXPOSE 80
