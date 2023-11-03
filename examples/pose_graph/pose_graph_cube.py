# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pathlib
import subprocess
from typing import List, Type, cast

import hydra
import omegaconf
import torch
from scipy.io import savemat

import theseus as th
import theseus.utils.examples as theg
from theseus.optimizer.linear.linear_solver import LinearSolver
from theseus.utils.examples.pose_graph.dataset import PoseGraphEdge

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
#   python examples/pose_graph_cube.py

# Logger
log = logging.getLogger(__name__)

DATASET_DIR = pathlib.Path.cwd() / "datasets" / "pose_graph" / "cube"

dtype = torch.float64


def get_batch_data(pg_batch: theg.PoseGraphDataset, pose_indices: List[int]):
    batch = {
        pg_batch.poses[index].name: pg_batch.poses[index].tensor
        for index in pose_indices
    }
    batch.update({pg_batch.poses[0].name + "__PRIOR": pg_batch.poses[0].tensor.clone()})
    batch.update(
        {edge.relative_pose.name: edge.relative_pose.tensor for edge in pg_batch.edges}
    )
    return batch


def run(
    cfg: omegaconf.OmegaConf,
    pg: theg.PoseGraphDataset,
    results_path: pathlib.Path,
    batch_size: int,
):
    pg.to(cfg.device)
    objective = th.Objective(dtype=dtype)

    pg_batch = pg.get_batch_dataset(0)
    pose_indices: List[int] = [index for index, _ in enumerate(pg_batch.poses)]

    for edge in pg_batch.edges:
        relative_pose_cost = th.Between(
            pg_batch.poses[edge.i],
            pg_batch.poses[edge.j],
            edge.relative_pose,
            edge.weight,
        )
        objective.add(relative_pose_cost)

    pose_prior_cost = th.Difference(
        var=pg_batch.poses[0],
        cost_weight=th.ScaleCostWeight(
            torch.tensor(cfg.inner_optim.reg_w, dtype=dtype, device=cfg.device)
        ),
        target=pg_batch.poses[0].copy(new_name=pg_batch.poses[0].name + "__PRIOR"),
    )

    objective.add(pose_prior_cost)

    linear_solver_cls: Type[LinearSolver] = cast(
        Type[LinearSolver],
        th.LUCudaSparseSolver
        if cast(str, cfg.solver_device) == "cuda"
        else th.CholmodSparseSolver,
    )
    optimizer = th.GaussNewton(
        objective.to(cfg.device),
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
        abs_err_tolerance=0,
        rel_err_tolerance=0,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=linear_solver_cls,
        vectorize=True,
    )

    def run_batch(batch_idx: int):
        log.info(f" ------------------- Batch {batch_idx} ------------------- ")
        pg_batch = pg.get_batch_dataset(batch_idx=batch_idx)
        theseus_inputs = get_batch_data(pg_batch, pose_indices)
        objective.update(input_tensors=theseus_inputs)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        torch.cuda.reset_peak_memory_stats()
        optimizer.optimize(verbose=True)
        end_event.record()

        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event)
        forward_mem = torch.cuda.max_memory_allocated() / 1048576
        log.info(f"Forward pass took {forward_time} ms.")
        log.info(f"Forward pass used {forward_mem} MBs.")

        return [forward_time, forward_mem]

    log.info(f" ******************** BATCH SIZE {batch_size} ******************** ")

    forward_times = []
    forward_mems = []

    results = omegaconf.OmegaConf.to_container(cfg)
    results["batch_size"] = batch_size

    for batch_idx in range(pg.num_batches):
        forward_time, forward_mem = run_batch(batch_idx)
        forward_times.append(forward_time)
        forward_mems.append(forward_mem)

        results["forward_time"] = forward_times
        results["forward_mem"] = forward_mems
        file = (
            f"pgo_cube_{cfg.solver_device}_{cfg.inner_optim.solver}_{cfg.num_poses}_"
            f"{cfg.dataset_size}_{batch_size}_{cfg.device}.mat"
        )
        savemat(file, results)


@hydra.main(config_path="../configs/pose_graph", config_name="pose_graph_cube")
def main(cfg):
    log.info((subprocess.check_output("lscpu", shell=True).strip()).decode())

    num_poses = cfg.num_poses

    poses: List[th.SE3] = []
    edges: List[PoseGraphEdge] = []

    for n in range(cfg.dataset_size):
        num_poses, poses_n, edges_n = theg.pose_graph.read_3D_g2o_file(
            (f"{DATASET_DIR}/{num_poses}_poses_0.2_cube_{n}.g2o"),
        )
        if len(poses) == 0:
            poses = poses_n
            edges = edges_n
        else:
            for pose, pose_n in zip(poses, poses_n):
                pose.tensor = torch.cat((pose.tensor, pose_n.tensor))

            for edge, edge_n in zip(edges, edges_n):
                edge.relative_pose.tensor = torch.cat(
                    (edge.relative_pose.tensor, edge_n.relative_pose.tensor)
                )

    # create (or load) dataset
    results_path = pathlib.Path(os.getcwd())

    for batch_size in [16, 256]:
        pg = theg.PoseGraphDataset(poses=poses, edges=edges, batch_size=batch_size)
        run(cfg, pg, results_path, batch_size)


if __name__ == "__main__":
    main()
