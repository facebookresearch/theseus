# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pathlib

import hydra
import torch
from scipy.io import savemat

import theseus as th
import theseus.utils.examples as theg

# To run this example, you will need the cube datasets available at
# https://dl.fbaipublicfiles.com/theseus/pose_graph_data.tar.gz
#
# The steps below should let you run the example.
# From the root project folder do:
#   mkdir datasets
#   cd datasets
#   cp your/path/pose_graph_data.tar.gz .
#   tar -xzvf pose_graph_data.tar.gz
#   cd ..
#   python examples/pose_graph_benchmark.py

# Logger
log = logging.getLogger(__name__)


DATASET_DIR = pathlib.Path.cwd() / "datasets" / "pose_graph"


@hydra.main(config_path="../configs/pose_graph", config_name="pose_graph_benchmark")
def main(cfg):
    dataset_name = cfg.dataset
    file_path = f"{DATASET_DIR}/{dataset_name}_init.g2o"
    dtype = eval(f"torch.{cfg.dtype}")

    _, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path, dtype=torch.float64)
    d = 3

    objective = th.Objective(torch.float64)

    for edge in edges:
        cost_func = th.Between(
            verts[edge.i],
            verts[edge.j],
            edge.relative_pose,
            edge.weight,
        )
        objective.add(cost_func)

    pose_prior = th.Difference(
        var=verts[0],
        cost_weight=th.ScaleCostWeight(torch.tensor(1e-6, dtype=torch.float64)),
        target=verts[0].copy(new_name=verts[0].name + "PRIOR"),
    )
    objective.add(pose_prior)

    optimizer = th.LevenbergMarquardt(
        objective.to(dtype),
        max_iterations=10,
        step_size=1,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True,
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    inputs = {var.name: var.tensor for var in verts}
    optimizer.objective.update(inputs)

    start_event.record()
    torch.cuda.reset_peak_memory_stats()
    optimizer.optimize(verbose=True)
    end_event.record()

    torch.cuda.synchronize()
    forward_time = start_event.elapsed_time(end_event)
    forward_mem = torch.cuda.max_memory_allocated() / 1048576
    log.info(f"Forward pass took {forward_time} ms.")
    log.info(f"Forward pass used {forward_mem} MBs.")

    results = {}
    results["objective"] = objective.error_metric().detach().cpu().numpy().sum()
    results["R"] = torch.cat(
        [pose.tensor[:, :, :d].detach().cpu() for pose in verts]
    ).numpy()
    results["t"] = torch.cat(
        [pose.tensor[:, :, d].detach().cpu() for pose in verts]
    ).numpy()

    savemat(dataset_name + ".mat", results)


if __name__ == "__main__":
    main()
