# Dockerfile for base container on which app container is based
# Responsible for installing software and other expensive operations that don't have to be changed often
FROM python:3.8.3-buster

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
