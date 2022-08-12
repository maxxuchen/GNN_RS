#!/bin/sh

# remove any previous environment
conda env remove -n ml4rs

# create the environment from the dependency file
conda env create -n ml4rs -f conda.yaml

conda activate ml4rs

# install pytorch with cuda 10.2
# conda install -y pytorch==1.9.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -y pytorch==1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# install pytorch geometric
conda install -y pytorch-geometric==1.7.2 -c rusty1s -c conda-forge

# install ray
pip install "ray[default]"

# install package
pip install -e .