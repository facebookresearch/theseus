Documentation for Theseus
========================================
Theseus is a library for differentiable nonlinear optimization built on PyTorch to 
support constructing various problems in robotics and vision as end-to-end 
differentiable architectures.

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
.. code-block:: bash
    
    pip install theseus-ai

If you are interested in contributing to ``theseus``, instead install

.. code-block:: bash

    git clone https://github.com/facebookresearch/theseus.git
    pip install -e ".[dev]"
    python -m pytest theseus
    
and follow the more detailed instructions in `CONTRIBUTING <https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md>`_.

By default, unit tests include tests for our CUDA extensions. You can add the option ``-m "not cudaext"`` to 
skip them when installing without CUDA support.


Tutorials
---------
See `tutorials <https://github.com/facebookresearch/theseus/blob/main/tutorials/>`_ and `examples <https://github.com/facebookresearch/theseus/blob/main/examples/>`_ to learn about the API and usage.

.. toctree::
   :maxdepth: 3
   :caption: API Documentation
