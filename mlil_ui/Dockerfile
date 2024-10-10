FROM python:3.12

WORKDIR /code

COPY code/ /code/

RUN pip install --upgrade --no-cache-dir -r /code/requirements.txt

EXPOSE 1122

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1122" ]
