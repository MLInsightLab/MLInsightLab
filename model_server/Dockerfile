FROM python:latest

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./db_utils.py /code/db_utils.py
COPY ./main.py /code/main.py

EXPOSE 4488

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4488", "--root-path", "/backend" ]
