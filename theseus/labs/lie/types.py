# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum


class ltype(Enum):
    SE3 = 0
    SO3 = 1
    tgt = 2


SE3 = ltype.SE3
SO3 = ltype.SO3
tgt = ltype.tgt
