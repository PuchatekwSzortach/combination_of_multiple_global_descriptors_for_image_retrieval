docker run -it --rm \
    -v $PWD:/app:delegated \
    -v $PWD/../../data:/data:delegated \
    -v /tmp/logs:/tmp/logs:delegated \
    puchatek_w_szortach/combination_of_multiple_global_descriptors:latest bash