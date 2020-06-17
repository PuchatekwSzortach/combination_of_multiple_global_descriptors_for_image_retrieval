docker run -it --rm \
    -v $PWD:/app:delegated \
    -v /Users/Kuba/Code/data:/data:delegated \
    puchatek_w_szortach/combination_of_multiple_global_descriptors:latest bash
