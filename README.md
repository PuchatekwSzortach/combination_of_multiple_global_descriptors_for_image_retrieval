# Combination of Multiple Global Descriptors for Image Retrieval

This repository contains a Tensorflow 2.2 implementation of image ranking model based on following research papers:
- [In Defense of the Triplet Loss for Person Re-Identification][in_defense_of_triplets_loss] - by Alexander Hermans, Lucas Beyer, and Bastian Leibe
- [Hard-aware point-to-set deep metric for person re-identification][hard_aware_point_to_set_loss] - by Rui Yu, Zhiyong Dou, Song Bai, Zhaoxiang Zhang, Yongchao Xu, and Xiang Bai
- [Combination of Multiple Global Descriptors for Image Retrieval][combination_of_multiple_global_descriptors] - by HeeJae Jun, Byungsoo Ko, Youngjoon Kim, Insik Kim, and Jongtack Kim

Two models are provided:
- `net.ml.ImagesSimilarityComputer` - a simple reference model, somewhat similar to model used in [In Defense of the Triplet Loss for Person Re-Identification][in_defense_of_triplets_loss]
- `net.ml.CGDImagesSimilarityComputer` - model using a combination of glabl descriptors, based on achitecture from [Combination of Multiple Global Descriptors for Image Retrieval][combination_of_multiple_global_descriptors] (CGD)

Two types of losses are provided:
- batch hard triplets loss - as described in [In Defense of the Triplet Loss for Person Re-Identification][in_defense_of_triplets_loss]
- hard aware point to set loss - as described in [Hard-aware point-to-set deep metric for person re-identification][hard_aware_point_to_set_loss]

### Implementation details

There are a some differences between official CGD architecture and our implementation. Authors of CGD paper modify backbone ResNet-50 network so there is no downsampling between stage 3 and stage 4, resulting in higher-resolution outputs from base network. We don't modify base network in any way.

We include script for training and evaluation on [Stanford University's Cars 196 Dataset][cars_196]. While CGD paper crops exact cars locations from raw images and then resizes results to fixed size, we instead first pad raw images to squares, and then resize to fixed size.

### Results

Results are based on [Stanford University's Cars 196 Dataset][cars_196].

Somewhat different from results reported in [Combination of Multiple Global Descriptors for Image Retrieval][combination_of_multiple_global_descriptors], the best results were obtained using model with SPoC (sum from channels) head only.
Adding MAC and GeM heads brings down accuracy by ~5%.

k | Recall at k
--- | ----
1 | 0.650
2 | 0.751
4 | 0.828
8 | 0.884

Image below shows representative ranking performance.
Each row starts with a query image, marked with a blue dot, followed by top 8 ranked images for that query. Images with same category as query image are marked with a green dot.
Validation set contains about 8,000 images, with, on average, about 80 images per category.

![Alt results][results_image]

### How to run

This project can be run in a docker container.
Building the docker container is a two stage process:
- building app_base container (`./docker/app_base.Dockerfile`)
- buiding app container (`./docker/app.Dockerfile`)

You can build containers manually with docker, but there are also [invoke][invoke] tasks provided:
- `invoke docker.build-app-base-container` - builds base container. Based on `tensorflow/tensorflow:2.2.0-gpu` container, downloads weights for base network, installs python requirements
- `invoke docker.build-app-container` - based on `app-base` container created with command above, creates user, paths, environment variables, mounts code

You can then start the container manually, or with provided `invoke docker.run` command that takes care of mounting paths for data volume. `invoke docker.run` asks for password to execute `sudo chmod` on data volume path, so that docker container has write permissions on host system. This is necessary because user inside docker container isn't root.
It's easy to modify code to change this behaviour if you don't need to access any outputs on host system.

Once inside container, following key `invoke` commands are available:
- analysis.analyze-model-performance           Analyze model performance
- ml.train                                     Train model
- visualize.visualize-data                     Visualize data
- visualize.visualize-predictions-on-batches   Visualize image similarity ranking predictions on a few batches of data
- visualize.visualize-predictions-on-dataset   Visualize image similarity ranking predictions on a few

Most commands accept `--config-path` argument that accepts a path pointing to a configuration file. Sample configuration file is provided at `./config.yaml`.

### How to extend

Should you want to use this code to train and predict on a different data than [Cars 196][cars_196], you would need to:
- provide your own data loaders for training and analyzing - please refer to `net.data.Cars196TrainingLoopDataLoader` and `net.data.Cars196AnalysisDataLoader` for sample implementations
- provide a yaml configuration file pointing to paths with your data - please refer to `config.yaml` for the expected format

### Honorable mentions
In addition to research papers listed above, following works were consulted during making of this project:
- Olivier Moindrot for his [post on triples loss][triplet_loss_blog] that helped me troubleshoot problem with training divergence when distance between two embeddings was 0
- leftthomas for his [PyTorch implementation of CGD][leftthomas_CGD] that I consulted for implementation of descriptors implementations

[in_defense_of_triplets_loss]: https://arxiv.org/abs/1703.07737
[hard_aware_point_to_set_loss]: https://arxiv.org/abs/1807.11206
[combination_of_multiple_global_descriptors]: https://arxiv.org/abs/1903.10663
[cars_196]: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
[triplet_loss_blog]: https://omoindrot.github.io/triplet-loss
[leftthomas_CGD]: https://github.com/leftthomas/CGD
[results_image]: images/sample_ranking_results.jpg
[invoke]: https://www.pyinvoke.org/
