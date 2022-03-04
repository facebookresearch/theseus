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


# Func can be SO2.rotate, SE2.transform_to, SO3.unrotate, etc., whose third argument
# populates the jacobians
def check_projection_for_rotate_and_transform(
    Group, Point, Func, batch_size, generator
):
    group = Group.rand(batch_size, generator=generator, dtype=torch.float64)
    point = Point.rand(batch_size, generator=generator, dtype=torch.float64)

    def func(g, p):
        return Func(Group(data=g), p).data

    jac_raw = torch.autograd.functional.jacobian(func, (group.data, point.data))
    jac = []
    _ = Func(group, point, jac)

    # Check dense jacobian matrices
    actual = [group.project(jac_raw[0]), point.project(jac_raw[1])]

    expected = [
        torch.zeros(batch_size, point.dof(), batch_size, group.dof()).double(),
        torch.zeros(batch_size, point.dof(), batch_size, point.dof()).double(),
    ]

    aux_id = torch.arange(batch_size)

    expected[0][aux_id, :, aux_id, :] = jac[0]
    expected[1][aux_id, :, aux_id, :] = jac[1]

    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])

    # Check sparse jacobian matrices
    actual = [
        group.project(jac_raw[0][aux_id, :, aux_id, :], is_sparse=True),
        point.project(jac_raw[1][aux_id, :, aux_id, :], is_sparse=True),
    ]

    expected = jac
    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])
