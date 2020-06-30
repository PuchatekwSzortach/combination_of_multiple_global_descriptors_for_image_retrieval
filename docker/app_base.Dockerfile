# Dockerfile for base container on which app container is based
# Responsible for installing software and other expensive operations that don't have to be changed often
FROM tensorflow/tensorflow:2.2.0-gpu

# Install a few necessary libs and apps
RUN apt update && apt install -y wget vim git

# Add user for the container
RUN useradd -u 1010 -ms /bin/bash app_user

# Download base tensorflow model to app_user's folder, change permission so he can use it
RUN mkdir -p /home/app_user/.keras/models && \
    wget https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
        -O /home/app_user/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 && \
    chown -R app_user:app_user /home/app_user/.keras

# Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
