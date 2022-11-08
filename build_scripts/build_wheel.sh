#bin/bash

# -----------------
# This script creates and runs a docker image for compiling a wheel
# to install theseus.
#
# To use this script, from root theesus folder run 
#    ./build_scripts/build_wheel.sh ROOT_DIR COMMIT CUDA_VERSION THESEUS_VERSION(optional) 
#
# ROOT_DIR: is the directory where the Dockerfile, tar.gz and .whl files will be stored
#   (under a new subdirectory named theseus_docker_3.9)
# COMMIT: is a theseus commit hash or tag (e.g., 0.1.3).
# CUDA_VERSION: the version of CUDA to use. We have tested 10.2, 11.3, 11.6, and 11.7.
#   You can also pass "cpu" to compile without CUDA extensions.
# THESEUS_VERSION: defaults to COMMIT, otherwise it must match the version in the commit.
#
#   For example
#    ./build_scripts/build_wheel.sh . 0.1.0 10.2
#   
#   will run and store results under ./theseus_docker_3.9
# -----------------

# Ensure that 3 or 4 arguments 
# (ROOT_DIR, COMMIT, CUDA_VERSION, THESEUS_VERSION - optional) are provided.
die () {
    echo >&2 "$@"
    exit 1
}
[ "$#" -eq 3 ] || [ "$#" -eq 4 ] || die "3 or 4 arguments required, $# provided"
ROOT_DIR=$1
COMMIT=$2
CUDA_VERSION=$3
TH_VERSION=${4:-${COMMIT}}


SUPPORTED_CUDA_VERSIONS="10.2 11.3 11.6 11.7"
CUDA_VERSION_IS_SUPPORTED=$(echo "cpu ${SUPPORTED_CUDA_VERSIONS}" | grep -w ${CUDA_VERSION})
[ "${CUDA_VERSION_IS_SUPPORTED}" ] || die "CUDA_VERSION must be one of (cpu ${SUPPORTED_CUDA_VERSIONS})"


CUDA_SUFFIX=$(echo ${CUDA_VERSION} | sed 's/[.]//g')

TORCH_VERSION='"torch>=1.13"'
if [[ ${CUDA_VERSION} == "cpu" ]]
then 
    DEVICE_TAG=cpu
    IMAGE_NAME="pytorch/manylinux-cuda102"
    ENABLE_CUDA=0
    BASPACHO_CUDA_ARGS="-DBASPACHO_USE_CUBLAS=0"
else
    DEVICE_TAG="cu${CUDA_SUFFIX}"
    IMAGE_NAME="pytorch/manylinux-cuda${CUDA_SUFFIX}"
    ENABLE_CUDA=1

    BASPACHO_CUDA_ARCHS="60;70;75"
    TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5"
    if [[ ${CUDA_VERSION} != "10.2" ]]
    then
        BASPACHO_CUDA_ARCHS="${BASPACHO_CUDA_ARCHS};80"
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};8.0"
    fi
    DEPRECATED_TORCH_CUDA="10.2 11.3"
    CUDA_IS_TORCH_DEPRECATED=$(echo "${DEPRECATED_TORCH_CUDA}" | grep -w ${CUDA_VERSION})
    if [[ ${CUDA_IS_TORCH_DEPRECATED} ]]
    then
        TORCH_VERSION='"torch<1.13"'
    fi

    BASPACHO_CUDA_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc -DBASPACHO_CUDA_ARCHS='${BASPACHO_CUDA_ARCHS}'"
fi

for PYTHON_VERSION in 3.9; do
    # Create dockerfile to build in manylinux container
    DOCKER_DIR=${ROOT_DIR}/theseus_docker_${PYTHON_VERSION}
    mkdir -p ${DOCKER_DIR}
    echo """# ----------------
    FROM ${IMAGE_NAME}

    # --- Install baspacho dependencies (cmake, BLAS)
    RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2-linux-x86_64.sh -O ~/cmake3.24.sh
    RUN mkdir /opt/cmake3.24
    RUN /bin/bash ~/cmake3.24.sh --prefix=/opt/cmake3.24 --skip-license
    RUN yum makecache
    RUN yum -y install openblas-static

    # --- Install baspacho
    RUN git clone https://github.com/facebookresearch/baspacho.git
    WORKDIR baspacho
    # Note: to use static BLAS the option is really BLA_STATIC (https://cmake.org/cmake/help/latest/module/FindBLAS.html)
    RUN /opt/cmake3.24/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBLA_STATIC=ON \
        ${BASPACHO_CUDA_ARGS} \
        -DBUILD_SHARED_LIBS=OFF \
        -DBASPACHO_BUILD_TESTS=OFF -DBASPACHO_BUILD_EXAMPLES=OFF
    RUN /opt/cmake3.24/bin/cmake --build build -- -j16
    WORKDIR ..

    # --- Install conda and environment
    ENV CONDA_DIR /opt/conda
    RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda
    ENV PATH \$CONDA_DIR/bin:\$PATH
    RUN conda create --name theseus python=${PYTHON_VERSION}
    RUN source activate theseus

    # --- Install torch
    ENV CUDA_HOME /usr/local/cuda-${CUDA_VERSION}
    RUN pip install ${TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/${DEVICE_TAG}

    # --- Install sparse suitesparse
    RUN conda install -c conda-forge suitesparse

    # --- Compile theseus wheel
    RUN pip install build wheel
    RUN git clone https://github.com/facebookresearch/theseus.git
    WORKDIR theseus
    RUN git fetch --all --tags
    RUN git checkout ${COMMIT} -b tmp_build
    CMD BASPACHO_ROOT_DIR=/baspacho THESEUS_FORCE_CUDA=${ENABLE_CUDA} TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' python3 -m build --no-isolation
    """ > ${DOCKER_DIR}/Dockerfile

    # Run the container
    cd ${DOCKER_DIR}
    echo $(pwd)
    DOCKER_NAME=theseus_${PYTHON_VERSION}
    sudo docker build -t "${DOCKER_NAME}_img" .
    sudo docker run --name ${DOCKER_NAME} ${DOCKER_NAME}_img

    # Copy the wheel to host
    CP_STR="cp"$(echo ${PYTHON_VERSION} | sed 's/[.]//g')
    DOCKER_WHL="theseus/dist/theseus_ai-${TH_VERSION}-${CP_STR}-${CP_STR}-linux_x86_64.whl"
    if [[ ${CUDA_VERSION} == "11.6" ]]
    then
        PLUS_CU_TAG=""  # 11.6 will be the pypi version, so don't add +cu116
    else
        PLUS_CU_TAG="+${DEVICE_TAG}"
    fi
    HOST_WHL="theseus_ai-${TH_VERSION}${PLUS_CU_TAG}-${CP_STR}-${CP_STR}-manylinux_2_17_x86_64.whl"

    sudo docker cp "${DOCKER_NAME}:theseus/dist/theseus-ai-${TH_VERSION}.tar.gz" "theseus-ai-${TH_VERSION}.tar.gz"
    sudo docker cp "${DOCKER_NAME}:${DOCKER_WHL}" ${HOST_WHL}
    sudo docker rm ${DOCKER_NAME}
    sudo docker image rm "${DOCKER_NAME}_img"
done
