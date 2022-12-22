Getting started
===============

Installation
------------

Prerequisites
^^^^^^^^^^^^^
- We *strongly* recommend you install ``theseus`` in a venv or conda environment with Python 3.7-3.9.
- Theseus requires ``torch`` installation. To install for your particular CPU/CUDA configuration, follow the instructions in the PyTorch `website <https://pytorch.org/get-started/locally/>`_.
- For GPU support, Theseus requires `nvcc <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`_ to compile custom CUDA operations. Make sure it matches the version used to compile pytorch with ``nvcc --version.`` If not, install it and ensure its location is on your system's ``$PATH`` variable.
- Theseus also requires `suitesparse <https://people.engr.tamu.edu/davis/suitesparse.html>`_, which you can install via:
    - ``sudo apt-get install libsuitesparse-dev`` (Ubuntu).
    - ``conda install -c conda-forge suitesparse`` (Mac).

Installing
^^^^^^^^^^

pypi
""""

.. code-block:: bash

    pip install theseus-ai

We currently provide wheels with our CUDA extensions compiled using CUDA 10.2 and Python 3.9.
For other CUDA versions, consider installing from source or using our 
`build script <https://github.com/facebookresearch/theseus/blob/main/build_scripts/build_wheel.sh>`_.

From source
"""""""""""
The simplest way to install Theseus from source is by running the following (see further below to also include BaSpaCho)

.. code-block:: bash

    git clone https://github.com/facebookresearch/theseus.git
    pip install -e .
    python -m pytest tests

If you are interested in contributing to ``theseus``, instead install using 

.. code-block:: bash

    pip install -e ".[dev]"

and follow the more detailed instructions in `CONTRIBUTING <https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md>`_.

**Installing BaSpaCho extensions from source**
By default, installing from source doesn't include our BaSpaCho sparse 
solver extension. For this, follow these steps:

1. Compile BaSpaCho from source following instructions `here <https://github.com/facebookresearch/baspacho>`_. We recommend using flags `-DBLA_STATIC=ON -DBUILD_SHARED_LIBS=OFF`.
2. Run 

    .. code-block:: bash

    git clone https://github.com/facebookresearch/theseus.git && cd theseus
    BASPACHO_ROOT_DIR=<path/to/root/baspacho/dir> pip install -e .

    where the BaSpaCho root dir must have binaries in the subdirectory `build`.

Unit tests
""""""""""
With ``dev`` installation, you can run unit tests via

.. code-block:: bash

    python -m pytest tests

By default, unit tests include tests for our CUDA extensions. You can add the option `-m "not cudaext"`
to skip them when installing without CUDA support. Additionally, the tests for sparse solver BaSpaCho are automatically 
skipped when its extlib is not compiled.

Tutorials
---------
See `tutorials <https://github.com/facebookresearch/theseus/blob/main/tutorials/>`_ and `examples <https://github.com/facebookresearch/theseus/blob/main/examples/>`_ to learn about the API and usage.
