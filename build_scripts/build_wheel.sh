#bin/bash

# Ensure that 2 arguments (ROOT_DIR, TAG) are provided
die () {
    echo >&2 "$@"
    exit 1
}
[ "$#" -eq 2 ] || die "2 arguments required, $# provided"
ROOT_DIR=$1
TAG=$2
# This needs to match the version in main on Github
THESEUS_VERSION=$(head theseus/__init__.py -n 6 | tail -n 1 | awk -F'"' '{print $2}')

for PYTHON_VERSION in 3.9; do
    for CUDA_VERSION in 10.2 11.6; do

        CUDA_SUFFIX=$(echo ${CUDA_VERSION} | sed 's/[.]//g')

        # Create dockerfile to build in manylinux container
        DOCKER_DIR=${ROOT_DIR}/docker_${PYTHON_VERSION}
        mkdir -p ${DOCKER_DIR}
        echo """FROM pytorch/manylinux-cuda${CUDA_SUFFIX}
        ENV CONDA_DIR /opt/conda
        RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
            /bin/bash ~/miniconda.sh -b -p /opt/conda

        ENV PATH \$CONDA_DIR/bin:\$PATH
        RUN conda create --name theseus python=${PYTHON_VERSION}
        RUN source activate theseus
        RUN which python
        ENV CUDA_HOME /usr/local/cuda-${CUDA_VERSION}
        RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu${CUDA_SUFFIX}
        RUN conda install -c conda-forge suitesparse
        RUN pip install build wheel
        RUN git clone https://github.com/facebookresearch/theseus.git
        WORKDIR theseus
        RUN git checkout ${TAG} -b tmp_build
        CMD python3 -m build --no-isolation
        """ > ${DOCKER_DIR}/Dockerfile

        # Run the container
        cd ${DOCKER_DIR}
        echo $(pwd)
        DOCKER_NAME=theseus_${PYTHON_VERSION}
        sudo docker build -t "${DOCKER_NAME}_img" .
        sudo docker run --gpus all --name ${DOCKER_NAME} ${DOCKER_NAME}_img

        # # Copy the wheel to host
        CP_STR="cp"$(echo ${PYTHON_VERSION} | sed 's/[.]//g')
        DOCKER_WHL="theseus/dist/theseus_ai-${THESEUS_VERSION}-${CP_STR}-${CP_STR}-linux_x86_64.whl"
        HOST_WHL="theseus_ai-${THESEUS_VERSION}+cu${CUDA_SUFFIX}-${CP_STR}-${CP_STR}-manylinux_2_17_x86_64.whl"
        sudo docker cp "${DOCKER_NAME}:${DOCKER_WHL}" ${DOCKER_DIR}/${HOST_WHL}

        sudo docker rm ${DOCKER_NAME}
        sudo docker image rm "${DOCKER_NAME}_img"
    done
done
