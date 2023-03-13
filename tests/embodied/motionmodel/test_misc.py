import theseus as th
import torch

from theseus.utils import check_jacobians


def test_nonholonomic():
    rng = torch.Generator()
    rng.manual_seed(0)
    # Check SE2 pose version
    pose = th.SE2.rand(10, generator=rng)
    vel = th.Vector.rand(10, 3, generator=rng)
    w = th.ScaleCostWeight(1.0)
    cf = th.eb.Nonholonomic(pose, vel, w)
    check_jacobians(cf, num_checks=100, tol=1e-5)
    # Check Vector3 pose version
    pose = th.Vector.rand(10, 3, generator=rng)
    cf = th.eb.Nonholonomic(pose, vel, w)
    check_jacobians(cf, num_checks=100, tol=1e-5)
