# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from torch import __version__ as _torch_version
from semantic_version import Version

if Version(_torch_version) < Version("2.0.0"):
    warnings.warn(
        "Using torch < 2.0 for theseus is deprecated and compatibility will be "
        "discontinued in future releases.",
        FutureWarning,
    )

__version__ = "0.2.0rc1"
