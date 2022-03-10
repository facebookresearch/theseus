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
    Group, Point, Func, batch_size, generator=None
):
    group = Group.rand(batch_size, generator=generator, dtype=torch.float64)
    point = Point.rand(batch_size, generator=generator, dtype=torch.float64)

    def func(g, p):
        return Func(Group(data=g), p).data

    jac_raw = torch.autograd.functional.jacobian(func, (group.data, point.data))
    jac = []

    # Check returns
    rets = Func(group, point, jac)
    assert torch.allclose(rets.data, func(group.data, point.data))

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


def check_projection_for_compose(Group, batch_size, generator=None):
    group1 = Group.rand(batch_size, generator=generator, dtype=torch.float64)
    group2 = Group.rand(batch_size, generator=generator, dtype=torch.float64)

    jac = []
    rets = group1.compose(group2, jacobians=jac)

    def func(g1, g2):
        return Group(data=g1).compose(Group(data=g2)).to_matrix()

    jac_raw = torch.autograd.functional.jacobian(func, (group1.data, group2.data))

    # Check returns
    assert torch.allclose(rets.to_matrix(), func(group1.data, group2.data))

    # Check for dense jacobian matrices
    temp = [group1.project(jac_raw[0]), group2.project(jac_raw[1])]
    actual = [
        torch.zeros(
            batch_size, rets.dof(), batch_size, rets.dof(), dtype=torch.float64
        ),
        torch.zeros(
            batch_size, rets.dof(), batch_size, rets.dof(), dtype=torch.float64
        ),
    ]

    for n in torch.arange(batch_size):
        for i in torch.arange(rets.dof()):
            actual[0][:, :, n, i] = rets.vee(
                rets.inverse().to_matrix() @ temp[0][:, :, :, n, i]
            )
            actual[1][:, :, n, i] = rets.vee(
                rets.inverse().to_matrix() @ temp[1][:, :, :, n, i]
            )

    expected = [
        torch.zeros(
            batch_size, rets.dof(), batch_size, rets.dof(), dtype=torch.float64
        ),
        torch.zeros(
            batch_size, rets.dof(), batch_size, rets.dof(), dtype=torch.float64
        ),
    ]

    aux_id = torch.arange(batch_size)

    expected[0][aux_id, :, aux_id, :] = jac[0]
    expected[1][aux_id, :, aux_id, :] = jac[1]

    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])

    # Check for sparse jacobian matrices
    temp = [
        group1.project(jac_raw[0][aux_id, :, :, aux_id], is_sparse=True),
        group2.project(jac_raw[1][aux_id, :, :, aux_id], is_sparse=True),
    ]
    actual = [
        torch.zeros(batch_size, rets.dof(), rets.dof(), dtype=torch.float64),
        torch.zeros(batch_size, rets.dof(), rets.dof(), dtype=torch.float64),
    ]

    for i in torch.arange(rets.dof()):
        actual[0][:, :, i] = rets.vee(rets.inverse().to_matrix() @ temp[0][:, :, :, i])
        actual[1][:, :, i] = rets.vee(rets.inverse().to_matrix() @ temp[1][:, :, :, i])

    expected = jac

    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])


def check_projection_for_inverse(Group, batch_size, generator=None):
    group = Group.rand(batch_size, generator=generator, dtype=torch.float64)

    jac = []
    rets = group.inverse(jacobian=jac)

    def func(g):
        return Group(data=g).inverse().to_matrix()

    jac_raw = torch.autograd.functional.jacobian(func, (group.data))

    # Check returns
    assert torch.allclose(rets.to_matrix(), func(group.data))

    # Check for dense jacobian matrices
    temp = group.project(jac_raw)
    actual = torch.zeros(
        batch_size, group.dof(), batch_size, group.dof(), dtype=torch.float64
    )

    for n in torch.arange(batch_size):
        for i in torch.arange(group.dof()):
            actual[:, :, n, i] = group.vee(group.to_matrix() @ temp[:, :, :, n, i])

    expected = torch.zeros(
        batch_size, group.dof(), batch_size, group.dof(), dtype=torch.float64
    )

    aux_id = torch.arange(batch_size)

    expected[aux_id, :, aux_id, :] = jac[0]

    assert torch.allclose(actual, expected)

    # Check for sparse jacobian matrices
    temp = group.project(jac_raw[aux_id, :, :, aux_id], is_sparse=True)
    actual = torch.zeros(batch_size, group.dof(), group.dof(), dtype=torch.float64)

    for i in torch.arange(group.dof()):
        actual[:, :, i] = group.vee(group.to_matrix() @ temp[:, :, :, i])

    expected = jac[0]

    assert torch.allclose(actual, expected)


def check_projection_for_exp_map(tangent_vector, Group, atol=1e-8):
    batch_size = tangent_vector.shape[0]
    dof = tangent_vector.shape[1]
    aux_id = torch.arange(batch_size)

    def exp_func(xi):
        return Group.exp_map(xi).to_matrix()

    actual = []
    group = Group.exp_map(tangent_vector, jacobians=actual).to_matrix()
    jac_raw = torch.autograd.functional.jacobian(exp_func, (tangent_vector.data))[
        aux_id, :, :, aux_id
    ]
    expected = torch.cat(
        [
            Group.vee(group.inverse() @ jac_raw[:, :, :, i]).view(-1, dof, 1)
            for i in torch.arange(dof)
        ],
        dim=2,
    )

    assert torch.allclose(actual[0], expected, atol=atol)


def check_projection_for_log_map(tangent_vector, Group, atol=1e-8):
    batch_size = tangent_vector.shape[0]
    aux_id = torch.arange(batch_size)
    group = Group.exp_map(tangent_vector)

    def log_func(group):
        return Group(data=group).log_map()

    jac_raw = torch.autograd.functional.jacobian(log_func, (group.data))
    expected = group.project(jac_raw[aux_id, :, aux_id], is_sparse=True)
    actual = []
    _ = group.log_map(jacobians=actual)

    assert torch.allclose(actual[0], expected, atol=atol)
