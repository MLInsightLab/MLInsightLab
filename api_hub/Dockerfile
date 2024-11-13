FROM python:3.12

# Change the workdir
WORKDIR /code

# Copy and install python requirements
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy and install main files
COPY ./db_utils.py /code/db_utils.py
COPY ./utils.py /code/utils.py
COPY ./main.py /code/main.py

# Add group mlil and add root to that group
RUN groupadd -g 1004 mlil
RUN adduser root mlil

# Expose necessary port
EXPOSE 4488

# Run the service
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4488", "--root-path", "/api" ]
