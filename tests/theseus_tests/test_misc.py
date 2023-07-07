# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from theseus._version import lt_version


def test_lt_version():
    assert not lt_version("2.0.0", "0.4.0")
    assert not lt_version("1.13.0abcd", "0.4.0")
    assert not lt_version("0.4.1+yzx", "0.4.0")
    assert lt_version("1.13.0.1.2.3.4", "2.0.0")
    assert lt_version("1.13.0.1.2+abc", "2.0.0")
    with pytest.raises(ValueError):
        lt_version("1.2", "0.4.0")
        lt_version("1", "0.4.0")
        lt_version("1.", "0.4.0")
