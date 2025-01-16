# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import warnings
from typing import Tuple

import torch


# Returns True/False if version string v1 is less than version string v2
def lt_version(v1: str, v2: str) -> bool:
    def _as_tuple(s: str) -> Tuple[int, int, int]:
        pattern = r"^[\d.]+"
        match = re.match(pattern, s)
        try:
            return tuple(int(x) for x in match.group().split(".")[:3])  # type: ignore
        except Exception:
            raise ValueError(
                f"String {s} cannot be converted to (mayor, minor, micro) format."
            )

    x1, y1, z1 = _as_tuple(v1)
    x2, y2, z2 = _as_tuple(v2)
    return x1 < x2 or (x1 == x2 and y1 < y2) or (x1 == x2 and y1 == y2 and z1 < z2)


if lt_version(torch.__version__, "2.0.0"):
    warnings.warn(
        "Using torch < 2.0 for theseus is deprecated and compatibility will be "
        "discontinued in future releases.",
        FutureWarning,
    )

__version__ = "0.2.3"
