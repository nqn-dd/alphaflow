# Original Copyright 2021 DeepMind Technologies Limited
# Modification Copyright 2022 # Copyright 2021 AlQuraishi Laboratory
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This container builds on the OpenFold container, and installs AlphaFlow + FastAPI service.
# Image size: ~25GB with ESMFlow weights baked in.
#
# OpenFold is quite difficult to get working, as it installs custom torch kernels.
# Adapted from https://github.com/aws-solutions-library-samples/aws-batch-arch-for-protein-folding

FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  wget \
  libxml2 \
  cuda-minimal-build-11-3 \
  libcusparse-dev-11-3 \
  libcublas-dev-11-3 \
  libcusolver-dev-11-3 \
  git \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get autoremove -y \
  && apt-get clean

RUN wget -q -P /tmp -O /tmp/miniconda.sh \
  "https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-$(uname -m).sh" \
  && bash /tmp/miniconda.sh -b -p /opt/conda \
  && rm /tmp/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN git clone https://github.com/aqlaboratory/openfold.git /opt/openfold \
  && cd /opt/openfold \
  && git checkout 1d878a1203e6d662a209a95f71b90083d5fc079c

# Downgrade pip to <24.1 so it accepts pytorch_lightning==1.5.10's invalid metadata
# (the .* version suffix in torch dependency spec was deprecated in pip 24.1)
RUN pip install "pip<24.1"

# Installing into the base environment since the container only runs AlphaFlow
RUN conda env update -n base --file /opt/openfold/environment.yml \
  && conda clean --all --force-pkgs-dirs --yes

RUN wget -q -P /opt/openfold/openfold/resources \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN patch -p0 -d /opt/conda/lib/python3.9/site-packages/ < /opt/openfold/lib/openmm.patch

# Install OpenFold
RUN cd /opt/openfold \
  && pip3 install --upgrade pip --no-cache-dir \
  && python3 setup.py install

# Install AlphaFlow from our fork
COPY alphaflow/ /opt/alphaflow/alphaflow/
COPY predict.py train.py /opt/alphaflow/
COPY splits/ /opt/alphaflow/splits/

# Install alphaflow packages (torch CUDA version must match container base)
RUN python -m pip install numpy==1.21.2 pandas==1.5.3 && \
    python -m pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html && \
    python -m pip install biopython==1.79 dm-tree==0.1.6 modelcif==0.7 ml-collections==0.1.0 scipy==1.7.3 absl-py einops && \
    python -m pip install pytorch_lightning==2.0.4 fair-esm mdtraj

# Install FastAPI service dependencies
COPY requirements-service.txt /opt/alphaflow/
RUN python -m pip install --no-cache-dir -r /opt/alphaflow/requirements-service.txt

# Copy service wrapper
COPY main.py /opt/alphaflow/

WORKDIR /opt/alphaflow

# Bake ESMFlow weights into the image (do not download at runtime)
RUN mkdir -p params && \
    wget -q -O params/esmflow_md_base_202402.pt \
      "https://alphaflow.s3.amazonaws.com/params/esmflow_md_base_202402.pt"

# Bake ESM2 weights (used by ESMFold backbone)
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q -O /root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt \
      https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt && \
    wget -q -O /root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D-contact-regression.pt \
      https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt

EXPOSE 8025

CMD ["python", "main.py"]
