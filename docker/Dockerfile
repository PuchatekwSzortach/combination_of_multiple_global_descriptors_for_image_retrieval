FROM python:3.8.3-buster

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./docker/bashrc /root/.bashrc

COPY . /app
