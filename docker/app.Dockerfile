# Dockerfile for app container
FROM python:3.8.3-buster

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN useradd -u 1010 -ms /bin/bash app_user
USER app_user

COPY ./docker/bashrc /home/app_user/.bashrc

ENV PYTHONPATH=.

COPY . /app
