# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

# Creating a new conda environment for argschema python bug
RUN conda create -n dispatcher python=3.8

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "dispatcher", "/bin/bash", "-c"]
RUN pip install -U --no-cache-dir \
    aind-codeocean-api==0.4.2 \
    python-dotenv==0.21.1 \
    aind-data-schema==1.0.0 \
    aind-ng-link==1.0.17 \
    python-dotenv==0.21.1 \
    smartsheet-dataframe==0.3.4

RUN conda install conda-forge::awscli
