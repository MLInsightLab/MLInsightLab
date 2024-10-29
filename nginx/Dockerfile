# Use the official nginx image as base
FROM nginx:latest

# Update
RUN apt update
RUN apt upgrade -y
RUN apt install apache2-utils -y

# Copy the custom nginx configuration file into the container
COPY nginx.nonssl.conf /etc/nginx/nginx.nonssl.conf
COPY nginx.ssl.conf /etc/nginx/nginx.ssl.conf

# Copy the password setup file to the container
COPY setup.sh /code/setup.sh

EXPOSE 80 443
