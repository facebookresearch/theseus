import pytest

from omegaconf import OmegaConf

import examples.pose_graph.pose_graph_synthetic as pgo


@pytest.fixture
def default_cfg():
    cfg = OmegaConf.load("examples/configs/pose_graph/pose_graph_synthetic.yaml")
    cfg.outer_optim.num_epochs = 1
    cfg.outer_optim.max_num_batches = 4
    cfg.batch_size = 16
    cfg.profile = False
    cfg.savemat = False
    cfg.inner_optim.optimizer_kwargs.verbose = False
    return cfg


def test_pgo_cpu_losses(default_cfg):
    default_cfg.inner_optim.solver = "dense"
    default_cfg.solver_device = "cpu"
    losses = pgo.run(default_cfg)

    expected_losses = [
        0.023993924212205704,
        0.012185513500756136,
        0.009521925414731194,
        0.015123319953872455,
    ]

    for loss, expected_loss in zip(losses[0], expected_losses):
        assert loss == pytest.approx(expected_loss)
