# Dockerfile for app container
FROM puchatek_w_szortach/combination_of_multiple_global_descriptors_base:2020.06.22.v1

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN useradd -u 1010 -ms /bin/bash app_user
USER app_user

COPY ./docker/bashrc /home/app_user/.bashrc

ENV PYTHONPATH=.

COPY . /app
