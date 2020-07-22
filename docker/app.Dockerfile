# Dockerfile for app container
FROM puchatek_w_szortach/combination_of_multiple_global_descriptors_base:2020.06.30.v1

# Update python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Setup bashrc for app user
COPY ./docker/bashrc /home/app_user/.bashrc

# Setup PYTHONPATH
ENV PYTHONPATH=.

# Tensorflow keeps on using deprecated APIs ^^
ENV PYTHONWARNINGS="ignore::DeprecationWarning:tensorflow"

# Select user container should be run with
USER app_user

# Set up working directory
WORKDIR /app

# Copy app code
COPY . /app
