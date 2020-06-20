# Dockerfile for base container on which app container is based
# Responsible for installing software and other expensive operations that don't have to be changed often
FROM puchatek_w_szortach/combination_of_multiple_global_descriptors_base:2020.06.20.v1

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
