# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import theseus as th

x = th.Point2()
y = th.Point2()

z1: th.Point2 = x + y  # noqa: F841
z2: th.Point2 = x - y  # noqa: F841
z3: th.Point2 = x * y  # noqa: F841
z4: th.Point2 = x / y  # noqa: F841
z5: th.Point2 = -x  # noqa: F841
z6: th.Point2 = x.cat(y)  # noqa: F841
z7: th.Point2 = x.abs()  # noqa: F841


x1 = th.Point3()
y1 = th.Point3()

w1: th.Point3 = x1 + y1  # noqa: F841
w2: th.Point3 = x1 - y1  # noqa: F841
w3: th.Point3 = x1 * y1  # noqa: F841
w4: th.Point3 = x1 / y1  # noqa: F841
w5: th.Point3 = -x1  # noqa: F841
w6: th.Point3 = x1.cat(y1)  # noqa: F841
w7: th.Point3 = x1.abs()  # noqa: F841
