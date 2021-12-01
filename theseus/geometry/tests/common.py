# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from theseus.constants import EPS
from theseus.utils import numeric_jacobian


def check_exp_map(tangent_vector, group_cls):
    group = group_cls.exp_map(tangent_vector)
    assert torch.allclose(
        group_cls.hat(tangent_vector).matrix_exp(),
        group.to_matrix(),
        atol=EPS,
    )


def check_log_map(tangent_vector, group_cls):
    assert torch.allclose(
        tangent_vector, group_cls.exp_map(tangent_vector).log_map(), atol=EPS
    )


def check_compose(group_1, group_2):
    Jcmp = []
    composition = group_1.compose(group_2, jacobians=Jcmp)
    expected_matrix = group_1.to_matrix() @ group_2.to_matrix()
    expected_jacs = numeric_jacobian(
        lambda groups: groups[0].compose(groups[1]),
        [group_1, group_2],
    )
    assert torch.allclose(composition.to_matrix(), expected_matrix, atol=EPS)
    assert torch.allclose(Jcmp[0], expected_jacs[0])
    assert torch.allclose(Jcmp[1], expected_jacs[1])


def check_inverse(group):
    tangent_vector = group.log_map()
    inverse_group = group.exp_map(-tangent_vector)
    jac = []
    inverse_result = group.inverse(jacobian=jac)
    expected_jac = numeric_jacobian(lambda groups: groups[0].inverse(), [group])
    assert torch.allclose(
        inverse_group.to_matrix(), inverse_result.to_matrix(), atol=EPS
    )
    assert torch.allclose(jac[0], expected_jac[0])


def check_adjoint(group, tangent_vector):
    group_matrix = group.to_matrix()
    tangent_left = group.__class__.adjoint(group) @ tangent_vector.unsqueeze(2)
    tangent_right = group.__class__.vee(
        group_matrix @ group.hat(tangent_vector) @ group.inverse().to_matrix()
    )
    assert torch.allclose(tangent_left.squeeze(2), tangent_right, atol=EPS)
