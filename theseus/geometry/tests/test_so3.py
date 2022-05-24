import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import EPS
from theseus.utils import numeric_jacobian

from .common import (
    check_adjoint,
    check_compose,
    check_exp_map,
    check_jacobian_for_local,
    check_projection_for_compose,
    check_projection_for_exp_map,
    check_projection_for_inverse,
    check_projection_for_log_map,
    check_projection_for_rotate_and_transform,
)


def check_SO3_log_map(tangent_vector):
    error = (tangent_vector - th.SO3.exp_map(tangent_vector).log_map()).norm(dim=1)
    error = torch.minimum(error, (error - 2 * np.pi).abs())
    assert torch.allclose(error, torch.zeros_like(error), atol=EPS)


def check_SO3_to_quaternion(so3: th.SO3, atol=1e-10):
    quaternions = so3.to_quaternion()
    assert torch.allclose(
        th.SO3(quaternion=quaternions).to_matrix(), so3.to_matrix(), atol=atol
    )


def test_exp_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in [1, 20, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        check_exp_map(tangent_vector, th.SO3)
        check_projection_for_exp_map(tangent_vector, th.SO3)

    # SO3.exp_map uses approximations for small theta
    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-5
        check_exp_map(tangent_vector, th.SO3)
        check_projection_for_exp_map(tangent_vector, th.SO3)

    # SO3.exp_map uses the exact exponential map for small theta
    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 3e-3
        check_exp_map(tangent_vector, th.SO3)
        check_projection_for_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-11
        check_exp_map(tangent_vector, th.SO3)
        check_projection_for_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 2 * np.pi - 1e-11
        check_exp_map(tangent_vector, th.SO3)
        check_projection_for_exp_map(tangent_vector, th.SO3)


def test_log_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        check_SO3_log_map(tangent_vector)
        check_projection_for_log_map(tangent_vector, th.SO3)

    # SO3.log_map uses approximations for small theta
    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-6
        check_SO3_log_map(tangent_vector)
        check_projection_for_log_map(tangent_vector, th.SO3)

    # SO3.log_map uses the exact logarithm map for small theta
    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-3
        check_SO3_log_map(tangent_vector)
        check_projection_for_log_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-11
        check_SO3_log_map(tangent_vector)
        check_projection_for_log_map(tangent_vector, th.SO3, 1e-7)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-3
        check_SO3_log_map(tangent_vector)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 2 * np.pi - 1e-11
        check_SO3_log_map(tangent_vector)


def test_quaternion():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-6
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-3
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-11
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-3
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 2 * np.pi - 1e-11
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so3 = th.SO3.rand(batch_size, generator=rng, dtype=torch.float64)
        tangent = torch.randn(batch_size, 3).double()
        check_adjoint(so3, tangent)


def test_compose():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so3_1 = th.SO3.rand(batch_size, generator=rng, dtype=torch.float64)
        so3_2 = th.SO3.rand(batch_size, generator=rng, dtype=torch.float64)
        check_compose(so3_1, so3_2)


def test_rotate_and_unrotate():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size_group in [1, 20, 100]:
            for batch_size_pnt in [1, 20, 100]:
                if (
                    batch_size_group != 1
                    and batch_size_pnt != 1
                    and batch_size_pnt != batch_size_group
                ):
                    continue

                so3 = th.SO3.rand(batch_size_group, generator=rng, dtype=torch.float64)
                point_tensor = torch.randn(batch_size_pnt, 3).double()

                jacobians_rotate = []
                rotated_point = so3.rotate(point_tensor, jacobians=jacobians_rotate)
                expected_rotated_data = so3.to_matrix() @ point_tensor.unsqueeze(2)
                jacobians_unrotate = []
                unrotated_point = so3.unrotate(rotated_point, jacobians_unrotate)

                # Check the operation result
                assert torch.allclose(
                    expected_rotated_data.squeeze(2), rotated_point.data, atol=EPS
                )
                assert torch.allclose(point_tensor, unrotated_point.data, atol=EPS)

                # Check the jacobians
                # function_dim = 3 because rotate(so3, (x, y, z)) --> (x_new, y_new, z_new)
                expected_jac = numeric_jacobian(
                    lambda groups: groups[0].rotate(groups[1]),
                    [so3, th.Point3(point_tensor)],
                    function_dim=3,
                )
                assert torch.allclose(jacobians_rotate[0], expected_jac[0])
                assert torch.allclose(jacobians_rotate[1], expected_jac[1])
                expected_jac = numeric_jacobian(
                    lambda groups: groups[0].unrotate(groups[1]),
                    [so3, rotated_point],
                    delta_mag=1e-5,
                    function_dim=3,
                )
                assert torch.allclose(jacobians_unrotate[0], expected_jac[0])
                assert torch.allclose(jacobians_unrotate[1], expected_jac[1])


def test_projection():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in [1, 20, 100]:
            # Test SO3.rotate
            check_projection_for_rotate_and_transform(
                th.SO3, th.Point3, th.SO3.rotate, batch_size, rng
            )

            # Test SO3.unrotate
            check_projection_for_rotate_and_transform(
                th.SO3, th.Point3, th.SO3.unrotate, batch_size, rng
            )

            # Test SO3.compose
            check_projection_for_compose(th.SO3, batch_size, rng)

            # Test SO3.inverse
            check_projection_for_inverse(th.SO3, batch_size, rng)


def test_local_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in [1, 20, 100]:
        group0 = th.SO3.rand(batch_size, dtype=torch.float64)
        group1 = th.SO3.rand(batch_size, dtype=torch.float64)

        check_jacobian_for_local(group0, group1, Group=th.SO3, is_projected=True)
