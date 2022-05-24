import numpy as np
import torch

import theseus.core as thcore
import theseus.embodied as thembod
import theseus.geometry as thgeom
from theseus.core.tests.common import (
    check_another_theseus_function_is_copy,
    check_another_theseus_tensor_is_copy,
)
from theseus.geometry.tests.test_se2 import create_random_se2
from theseus.utils import numeric_jacobian


def test_copy_moving_frame_between():
    f1 = thgeom.SE2()
    f2 = thgeom.SE2()
    p1 = thgeom.SE2()
    p2 = thgeom.SE2()
    meas = thgeom.SE2()
    cost_function = thembod.MovingFrameBetween(
        f1, f2, p1, p2, thcore.ScaleCostWeight(1.0), meas, name="name"
    )
    cost_function2 = cost_function.copy(new_name="new_name")
    check_another_theseus_function_is_copy(
        cost_function, cost_function2, new_name="new_name"
    )
    check_another_theseus_tensor_is_copy(cost_function2.frame1, f1)
    check_another_theseus_tensor_is_copy(cost_function2.frame2, f2)
    check_another_theseus_tensor_is_copy(cost_function2.pose1, p1)
    check_another_theseus_tensor_is_copy(cost_function2.pose2, p2)
    check_another_theseus_tensor_is_copy(cost_function2.measurement, meas)
    check_another_theseus_function_is_copy(
        cost_function.weight,
        cost_function2.weight,
        new_name=f"{cost_function.weight.name}_copy",
    )
    assert cost_function2.name == "new_name"


def test_jacobian_moving_frame_between():
    rng = torch.Generator()
    rng.manual_seed(0)
    cost_weight = thcore.ScaleCostWeight(1)
    for batch_size in [1, 10, 100]:
        f1 = create_random_se2(batch_size, rng)
        f2 = create_random_se2(batch_size, rng)
        p1 = create_random_se2(batch_size, rng)
        p2 = create_random_se2(batch_size, rng)
        measurement = create_random_se2(batch_size, rng)
        cost_function = thembod.MovingFrameBetween(
            f1, f2, p1, p2, cost_weight, measurement
        )

        def new_error_fn(groups):
            new_cost_function = thembod.MovingFrameBetween(
                groups[0], groups[1], groups[2], groups[3], cost_weight, measurement
            )
            return new_cost_function.measurement.retract(new_cost_function.error())

        expected_jacs = numeric_jacobian(new_error_fn, [f1, f2, p1, p2])
        jacobians, error_jac = cost_function.jacobians()
        error = cost_function.error()
        assert torch.allclose(error_jac, error)
        for i in range(4):
            assert torch.allclose(jacobians[i], expected_jacs[i], atol=1e-8)


def test_error_moving_frame_between_se2():

    measurement = thgeom.SE2(
        x_y_theta=torch.DoubleTensor([0.0, 0.0, np.pi / 6]).unsqueeze(0)
    )
    cost_weight = thcore.ScaleCostWeight(1)

    inputs = {
        "f1": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, np.pi / 2],
            ]
        ),
        "f2": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, np.pi / 2],
                [1.0, 1.0, np.pi / 4],
                [1.0, 1.0, np.pi / 4],
                [0.0, 0.0, 0.0],
            ]
        ),
        "p1": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, np.pi / 2],
                [1.0, 1.0, np.pi / 4],
                [2.0, 2.0, np.pi / 4],
                [2.0, 2.0, np.pi / 4],
            ]
        ),
        "p2": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, np.pi],
                [2.0, 2.0, np.pi / 2],
                [3.0, 3.0, np.pi / 2],
                [3.0, 3.0, np.pi / 2],
            ]
        ),
    }

    outputs = {
        "error": torch.DoubleTensor(
            [
                [0.0, 0.0, -0.52359878],
                [0.0, 0.0, -0.52359878],
                [-0.66650618, -0.86860776, -0.52359878],
                [-1.33301235, -1.73721552, -0.52359878],
                [4.64499601, 2.25899128, 1.83259571],
            ]
        )
    }
    n_tests = outputs["error"].shape[0]
    for i in range(0, n_tests):
        f1 = thgeom.SE2(x_y_theta=(inputs["f1"][i, :]).unsqueeze(0))
        f2 = thgeom.SE2(x_y_theta=(inputs["f2"][i, :]).unsqueeze(0))
        p1 = thgeom.SE2(x_y_theta=(inputs["p1"][i, :]).unsqueeze(0))
        p2 = thgeom.SE2(x_y_theta=(inputs["p2"][i, :]).unsqueeze(0))

        cost_fn = thembod.MovingFrameBetween(f1, f2, p1, p2, cost_weight, measurement)

        actual = cost_fn.error()
        expected = outputs["error"][i, :]

        assert np.allclose(actual.squeeze().numpy(), expected.squeeze().numpy())
