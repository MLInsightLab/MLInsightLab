FROM python:3.12

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip dask[complete] msgpack toolz

# Expose Dask's default port
EXPOSE 8786
