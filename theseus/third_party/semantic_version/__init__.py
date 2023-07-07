# -*- coding: utf-8 -*-
# Copyright (c) The python-semanticversion project
# This code is distributed under the two-clause BSD License.
# type: ignore

from .base import NpmSpec, SimpleSpec, Spec, SpecItem, Version, compare, match, validate

__author__ = "Raphaël Barrois <raphael.barrois+semver@polytechnique.org>"
try:
    # Python 3.8+
    from importlib.metadata import version

    __version__ = version("semantic_version")
except ImportError:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("semantic_version").version
