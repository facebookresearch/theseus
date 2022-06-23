# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from theseus.constants import TEST_EPS
from theseus.utils import numeric_jacobian


def check_exp_map(tangent_vector, group_cls, atol=TEST_EPS):
    group = group_cls.exp_map(tangent_vector)
    tangent_vector_double = tangent_vector.double()
    tangent_vector_double.to(dtype=torch.float64)
    assert torch.allclose(
        group_cls.hat(tangent_vector_double).matrix_exp(),
        group.to_matrix().double(),
        atol=atol,
    )


def check_log_map(tangent_vector, group_cls, atol=TEST_EPS):
    assert torch.allclose(
        tangent_vector, group_cls.exp_map(tangent_vector).log_map(), atol=atol
    )


def check_compose(group_1, group_2):
    Jcmp = []
    composition = group_1.compose(group_2, jacobians=Jcmp)
    expected_matrix = group_1.to_matrix() @ group_2.to_matrix()
    group_1_double = group_1.copy()
    group_1_double.to(torch.float64)
    group_2_double = group_2.copy()
    group_2_double.to(torch.float64)
    expected_jacs = numeric_jacobian(
        lambda groups: groups[0].compose(groups[1]),
        [group_1_double, group_2_double],
    )
    assert torch.allclose(composition.to_matrix(), expected_matrix, atol=TEST_EPS)
    assert torch.allclose(Jcmp[0].double(), expected_jacs[0], atol=TEST_EPS)
    assert torch.allclose(
        Jcmp[1].double(),
        torch.eye(6, 6, dtype=torch.float64)
        .unsqueeze(0)
        .expand(group_1.shape[0], 6, 6),
        atol=TEST_EPS,
    )
    if group_1.dtype == torch.float64:
        assert torch.allclose(Jcmp[1].double(), expected_jacs[1], atol=TEST_EPS)


def check_inverse(group):
    tangent_vector = group.log_map()
    inverse_group = group.exp_map(-tangent_vector.double())
    jac = []
    inverse_result = group.inverse(jacobian=jac)
    group_double = group.copy()
    group_double.to(torch.float64)
    expected_jac = numeric_jacobian(lambda groups: groups[0].inverse(), [group_double])
    assert torch.allclose(
        inverse_group.to_matrix().double(),
        inverse_result.to_matrix().double(),
        atol=TEST_EPS,
    )
    assert torch.allclose(jac[0].double(), expected_jac[0], atol=TEST_EPS)


def check_adjoint(group, tangent_vector):
    tangent_left = group.__class__.adjoint(group) @ tangent_vector.unsqueeze(2)
    group_matrix = group.to_matrix()
    tangent_right = group.__class__.vee(
        group_matrix.double()
        @ group.hat(tangent_vector.double())
        @ group.inverse().to_matrix().double()
    )
    assert torch.allclose(
        tangent_left.double().squeeze(2), tangent_right, atol=TEST_EPS
    )


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

    assert torch.allclose(actual[0], expected[0], atol=TEST_EPS)
    assert torch.allclose(actual[1], expected[1], atol=TEST_EPS)

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


def check_projection_for_exp_map(tangent_vector, Group, is_projected=True, atol=1e-8):
    batch_size = tangent_vector.shape[0]
    dof = tangent_vector.shape[1]
    aux_id = torch.arange(batch_size)

    def exp_func(xi):
        return Group.exp_map(xi).to_matrix()

    actual = []
    _ = Group.exp_map(tangent_vector, jacobians=actual).to_matrix()

    tangent_vector_double = tangent_vector.double()
    group_double = Group.exp_map(tangent_vector_double).to_matrix()
    jac_raw = torch.autograd.functional.jacobian(exp_func, (tangent_vector_double))

    if is_projected:
        jac_raw = jac_raw[aux_id, :, :, aux_id]
        expected = torch.cat(
            [
                Group.vee(group_double.inverse() @ jac_raw[:, :, :, i]).view(-1, dof, 1)
                for i in torch.arange(dof)
            ],
            dim=2,
        )
    else:
        expected = jac_raw[aux_id, :, aux_id]

    assert torch.allclose(actual[0].double(), expected, atol=atol)


def check_projection_for_log_map(tangent_vector, Group, is_projected=True, atol=1e-8):
    batch_size = tangent_vector.shape[0]
    aux_id = torch.arange(batch_size)
    group_double = Group.exp_map(tangent_vector.double())
    group = group_double.copy()
    if tangent_vector.dtype == torch.float32:
        group.data = group.data.float()

    def log_func(group):
        return Group(data=group).log_map()

    jac_raw = torch.autograd.functional.jacobian(log_func, (group_double.data))[
        aux_id, :, aux_id
    ]

    if is_projected:
        expected = group_double.project(jac_raw, is_sparse=True)
    else:
        expected = jac_raw

    actual = []
    _ = group.log_map(jacobians=actual)

    assert torch.allclose(actual[0].double(), expected, atol=atol)


def check_jacobian_for_local(group0, group1, Group, is_projected=True, atol=1e-8):
    batch_size = group0.shape[0]
    aux_id = torch.arange(batch_size)

    def local_func(group0, group1):
        return Group(data=group0).local(Group(data=group1))

    jac_raw = torch.autograd.functional.jacobian(local_func, (group0.data, group1.data))
    if is_projected:
        expected = [
            group0.project(jac_raw[0][aux_id, :, aux_id], is_sparse=True),
            group1.project(jac_raw[1][aux_id, :, aux_id], is_sparse=True),
        ]
    else:
        expected = [jac_raw[0][aux_id, :, aux_id], jac_raw[1][aux_id, :, aux_id]]

    actual = []
    _ = group0.local(group1, jacobians=actual)

    assert torch.allclose(actual[0], expected[0], atol=atol)
    assert torch.allclose(actual[1], expected[1], atol=atol)
