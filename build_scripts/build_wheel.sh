#bin/bash

# -----------------
# This script creates and runs a docker image for compiling a wheel
# to install theseus.
#
# To use this script, from root theesus folder run 
#    ./build_scripts/build_wheel.sh ROOT_DIR TAG CUDA_VERSION
#
# ROOT_DIR: is the directory where the Dockerfile, tar.gz and .whl files will be stored
#   (under a new subdirectory named theseus_docker_3.9)
# TAG: is a theseus tag (e.g., 0.1.0)
# CUDA_VERSION: the version of CUDA to use. We have tested 10.2, 11.3, and 11.6.
#   You can also pass "cpu" to compile without CUDA extensions. 
#
#   For example
#    ./build_scripts/build_wheel.sh . 0.1.0 10.2
#   
#   will run and store results under ./theseus_docker_3.9
# -----------------

# Ensure that 3 arguments (ROOT_DIR, TAG, CUDA_VERSION) are provided
die () {
    echo >&2 "$@"
    exit 1
}
[ "$#" -eq 3 ] || die "3 arguments required, $# provided"
ROOT_DIR=$1
TAG=$2
CUDA_VERSION=$3

CUDA_VERSION_SUPPORTED=$(echo "cpu 10.2 11.3 11.6" | grep -w ${CUDA_VERSION})
[ "${CUDA_VERSION_SUPPORTED}" ] || die "CUDA_VERSION must be one of (cpu, 10.2, 11.3, 11.6)"


CUDA_SUFFIX=$(echo ${CUDA_VERSION} | sed 's/[.]//g')

if [[ ${CUDA_VERSION} == "cpu" ]] 
then
    DEVICE_TAG=cpu
    IMAGE_NAME="pytorch/manylinux-cuda102"
    GPU_ARGS=""
else
    DEVICE_TAG="cu${CUDA_SUFFIX}"
    IMAGE_NAME="pytorch/manylinux-cuda${CUDA_SUFFIX}"
    GPU_ARGS=" --gpus all"
fi

for PYTHON_VERSION in 3.9; do
    # Create dockerfile to build in manylinux container
    DOCKER_DIR=${ROOT_DIR}/theseus_docker_${PYTHON_VERSION}
    mkdir -p ${DOCKER_DIR}
    echo """FROM ${IMAGE_NAME}
    ENV CONDA_DIR /opt/conda
    RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda
    ENV PATH \$CONDA_DIR/bin:\$PATH
    RUN conda create --name theseus python=${PYTHON_VERSION}
    RUN source activate theseus
    RUN which python
    ENV CUDA_HOME /usr/local/cuda-${CUDA_VERSION}
    RUN pip install torch --extra-index-url https://download.pytorch.org/whl/${DEVICE_TAG}
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
    sudo docker run${GPU_ARGS} --name ${DOCKER_NAME} ${DOCKER_NAME}_img

    # Copy the wheel to host
    CP_STR="cp"$(echo ${PYTHON_VERSION} | sed 's/[.]//g')
    if [[ ${CUDA_VERSION} == "cpu" ]] 
    then
        DOCKER_WHL="theseus/dist/theseus_ai-${TAG}-py3-none-any.whl"
        HOST_WHL="theseus_ai-${TAG}-py3-none-any.whl"
    else
        DOCKER_WHL="theseus/dist/theseus_ai-${TAG}-${CP_STR}-${CP_STR}-linux_x86_64.whl"
        if [[ ${CUDA_VERSION} == "10.2" ]]
        then
            PLUS_CU_TAG=""  # 10.2 will be the pypi version, so don't add +cu102
        else
            PLUS_CU_TAG="+${DEVICE_TAG}"
        fi
        HOST_WHL="theseus_ai-${TAG}${PLUS_CU_TAG}-${CP_STR}-${CP_STR}-manylinux_2_17_x86_64.whl"
    fi
    
    sudo docker cp "${DOCKER_NAME}:theseus/dist/theseus-ai-${TAG}.tar.gz" "${DOCKER_DIR}/theseus-ai-${TAG}.tar.gz"
    sudo docker cp "${DOCKER_NAME}:${DOCKER_WHL}" ${DOCKER_DIR}/${HOST_WHL}
    sudo docker rm ${DOCKER_NAME}
    sudo docker image rm "${DOCKER_NAME}_img"
done
