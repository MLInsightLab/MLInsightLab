FROM python:3.12

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip dask[complete] msgpack toolz

# Add group mlil and add root to that group
RUN groupadd -g 1004 mlil
RUN adduser root mlil

# Expose Dask's default port
EXPOSE 8786
