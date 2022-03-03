# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import nox


@nox.session()
def lint(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements/dev.txt")
    session.run("flake8", "theseus")
    session.run("black", "--check", "theseus")


@nox.session()
def mypy_and_tests(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("torch")
    session.install("-r", "requirements/dev.txt")
    session.run("mypy", "theseus")
    session.install("-e", ".")
    session.run("pytest", "theseus", "-m", "not cuda")
