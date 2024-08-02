FROM python:latest

WORKDIR /code

COPY code/ /code/

RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir -r /code/requirements.txt

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1122" ]
