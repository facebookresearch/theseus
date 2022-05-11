# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import os
import pathlib
import pstats

import random
from typing import Union, Dict, List, Type, cast

import hydra
import numpy as np
import omegaconf
import torch

import theseus as th
import theseus.utils.examples as theg
from theseus.optimizer.linearization import Linearization
from theseus.optimizer.linear import LinearSolver

import cProfile
import io
import subprocess

from scipy.io import savemat

BACKWARD_MODE = {
    "implicit": th.BackwardMode.IMPLICIT,
    "full": th.BackwardMode.FULL,
    "truncated": th.BackwardMode.TRUNCATED,
}

LINEARIZATION_MODE: Dict[str, Type[Linearization]] = {
    "sparse": th.SparseLinearization,
    "dense": th.DenseLinearization,
}

LINEAR_SOLVER_MODE: Dict[str, Type[LinearSolver]] = {
    "sparse": th.LUCudaSparseSolver,
    "dense": th.CholeskyDenseSolver,
}

# Smaller values} result in error
th.SO3.SO3_EPS = 1e-6

# Logger
log = logging.getLogger(__name__)


def print_histogram(
    pg: theg.PoseGraphDataset, var_dict: Dict[str, torch.Tensor], msg: str
):
    log.info(msg)
    with torch.no_grad():
        poses = [
            th.SE3(data=var_dict[pose.name], requires_check=False) for pose in pg.poses
        ]
        histogram = theg.pg_histogram(poses=poses, edges=pg.edges)
    for line in histogram.split("\n"):
        log.info(line)


def get_batch_data(
    pg_batch: theg.PoseGraphDataset, pose_indices: List[int], gt_pose_indices: List[int]
):
    batch = {
        pg_batch.poses[index].name: pg_batch.poses[index].data for index in pose_indices
    }
    batch.update({pg_batch.poses[0].name + "__PRIOR": pg_batch.poses[0].data.clone()})
    batch.update(
        {
            pg_batch.gt_poses[index].name: pg_batch.gt_poses[index].data
            for index in gt_pose_indices
        }
    )
    batch.update(
        {edge.relative_pose.name: edge.relative_pose.data for edge in pg_batch.edges}
    )
    return batch


def pose_loss(
    pose_vars: Union[List[th.SE2], List[th.SE3]],
    gt_pose_vars: Union[List[th.SE2], List[th.SE3]],
) -> torch.Tensor:
    loss: torch.Tensor = torch.zeros(
        1, dtype=pose_vars[0].dtype, device=pose_vars[0].device
    )
    poses_batch = th.SE3(
        data=torch.cat([pose.data for pose in pose_vars]), requires_check=False
    )
    gt_poses_batch = th.SE3(
        data=torch.cat([gt_pose.data for gt_pose in gt_pose_vars]), requires_check=False
    )
    pose_loss = th.local(poses_batch, gt_poses_batch).norm(dim=1)
    loss += pose_loss.sum()
    return loss


def run(
    cfg: omegaconf.OmegaConf, pg: theg.PoseGraphDataset, results_path: pathlib.Path
):
    device = torch.device("cuda")
    dtype = torch.float64
    pr = cProfile.Profile()

    BACKWARD_MODE = {
        "implicit": th.BackwardMode.IMPLICIT,
        "full": th.BackwardMode.FULL,
        "truncated": th.BackwardMode.TRUNCATED,
    }

    LINEARIZATION_MODE: Dict[str, Type[Linearization]] = {
        "sparse": th.SparseLinearization,
        "dense": th.DenseLinearization,
    }

    LINEAR_SOLVER_MODE: Dict[str, Type[LinearSolver]] = {
        "sparse": cast(
            Type[LinearSolver],
            th.LUCudaSparseSolver
            if cast(str, cfg.solver_device) == "cuda"
            else th.CholmodSparseSolver,
        ),
        "dense": th.CholeskyDenseSolver,
    }

    pg.to(device=device)

    with torch.no_grad():
        pose_loss_ref = pose_loss(pg.poses, pg.gt_poses).item()
    log.info(f"POSE LOSS (no learning):  {pose_loss_ref: .3f}")

    # Create the objective
    pg_batch = pg.get_batch_dataset(0)

    log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=dtype)
    robust_loss = th.WelschLoss(log_loss_radius=log_loss_radius, name="welsch_loss")

    objective = th.Objective(dtype=torch.float64)

    pose_indices: List[int] = [index for index, _ in enumerate(pg_batch.poses)]
    gt_pose_indices: List[int] = []

    forward_times = []
    backward_times = []
    forward_mems = []
    backward_mems = []

    for edge in pg_batch.edges:
        relative_pose_cost = th.eb.Between(
            pg_batch.poses[edge.i],
            pg_batch.poses[edge.j],
            edge.weight,
            edge.relative_pose,
            loss_function=robust_loss,
        )
        objective.add(relative_pose_cost, use_batches=cfg.inner_optim.use_batches)

    if cfg.inner_optim.regularize:
        pose_prior_cost = th.eb.VariableDifference(
            var=pg_batch.poses[0],
            cost_weight=th.ScaleCostWeight(
                torch.tensor(cfg.inner_optim.reg_w, dtype=dtype)
            ),
            target=pg_batch.poses[0].copy(new_name=pg_batch.poses[0].name + "__PRIOR"),
        )
        objective.add(pose_prior_cost, use_batches=cfg.inner_optim.use_batches)

    if cfg.inner_optim.ratio_known_poses > 0.0:
        pose_prior_weight = th.ScaleCostWeight(100 * torch.ones(1, dtype=dtype))
        for i in range(len(pg_batch.poses)):
            if np.random.rand() > cfg.inner_optim.ratio_known_poses:
                continue
            objective.add(
                th.eb.VariableDifference(
                    pg_batch.poses[i],
                    pose_prior_weight,
                    pg_batch.gt_poses[i],
                    name=f"pose_diff_{i}",
                ),
                use_batches=cfg.inner_optim.use_batches,
            )
            gt_pose_indices.append(i)

    pose_vars: List[th.SE3] = [
        cast(th.SE3, objective.optim_vars[pose.name]) for pose in pg_batch.poses
    ]

    # Create optimizer
    optimizer_cls: Type[th.NonlinearLeastSquares] = getattr(
        th, cfg.inner_optim.optimizer_cls
    )

    optimizer = optimizer_cls(
        objective,
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
        linearization_cls=LINEARIZATION_MODE[cast(str, cfg.inner_optim.solver)],
        linear_solver_cls=LINEAR_SOLVER_MODE[cast(str, cfg.inner_optim.solver)],
    )

    # Set up Theseus layer
    theseus_optim = th.TheseusLayer(optimizer)
    theseus_optim.to(device=device)

    # Outer optimization loop
    log_loss_radius_tensor = torch.nn.Parameter(
        torch.tensor([[3.0]], device=device, dtype=dtype)
    )
    model_optimizer = torch.optim.Adam([log_loss_radius_tensor], lr=cfg.outer_optim.lr)

    num_epochs = cfg.outer_optim.num_epochs

    def run_batch(batch_idx: int):
        log.info(f" ------------------- Batch {batch_idx} ------------------- ")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        pg_batch = pg.get_batch_dataset(batch_idx=batch_idx)
        theseus_inputs = get_batch_data(pg_batch, pose_indices, gt_pose_indices)
        theseus_inputs["log_loss_radius"] = log_loss_radius_tensor.clone()

        with torch.no_grad():
            pose_loss_ref = pose_loss(pg_batch.poses, pg_batch.gt_poses)

        start_event.record()
        torch.cuda.reset_peak_memory_stats()
        pr.enable()
        theseus_outputs, _ = theseus_optim.forward(
            input_data=theseus_inputs,
            optimizer_kwargs={
                "verbose": cfg.inner_optim.verbose,
                "track_err_history": cfg.inner_optim.track_err_history,
                "backward_mode": BACKWARD_MODE[cfg.inner_optim.backward_mode],
                "__keep_final_step_size__": cfg.inner_optim.keep_step_size,
                "grouped_retract": cfg.inner_optim.use_batches,
            },
        )
        pr.disable()
        end_event.record()

        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event)
        forward_mem = torch.cuda.max_memory_allocated() / 1048576
        log.info(f"Forward pass took {forward_time} ms.")
        log.info(f"Forward pass used {forward_mem} MBs.")

        start_event.record()
        torch.cuda.reset_peak_memory_stats()
        pr.enable()
        model_optimizer.zero_grad()
        loss = (pose_loss(pose_vars, pg_batch.gt_poses) - pose_loss_ref) / pose_loss_ref
        loss.backward()
        model_optimizer.step()
        backward_mem = torch.cuda.max_memory_allocated() / 1048576
        pr.disable()
        end_event.record()

        torch.cuda.synchronize()
        backward_time = start_event.elapsed_time(end_event)
        log.info(f"Backward pass took {backward_time} ms.")
        log.info(f"Backward pass used {backward_mem} MBs.")

        loss_value = torch.sum(loss.detach()).item()
        log.info(
            f"Loss: {loss_value} "
            f"Kernel Radius: exp({log_loss_radius_tensor.data.item()})="
            f"{torch.exp(log_loss_radius_tensor.data).item()}"
        )

        print_histogram(pg_batch, theseus_outputs, "Output histogram:")

        return [forward_time, backward_time, forward_mem, backward_mem]

    for epoch in range(num_epochs):
        log.info(f" ******************* EPOCH {epoch} ******************* ")

        forward_time_epoch = []
        backward_time_epoch = []
        forward_mem_epoch = []
        backward_mem_epoch = []

        for batch_idx in range(pg.num_batches):
            if batch_idx == cfg.outer_optim.max_num_batches:
                break
            forward_time, backward_time, forward_mem, backward_mem = run_batch(
                batch_idx
            )

            forward_time_epoch.append(forward_time)
            backward_time_epoch.append(backward_time)
            forward_mem_epoch.append(forward_mem)
            backward_mem_epoch.append(backward_mem)

        forward_times.append(forward_time_epoch)
        backward_times.append(backward_time_epoch)
        forward_mems.append(forward_mem_epoch)
        backward_mems.append(backward_mem_epoch)

        results = omegaconf.OmegaConf.to_container(cfg)
        results["forward_time"] = forward_times
        results["backward_time"] = backward_times
        results["forward_mem"] = forward_mems
        results["backward_mem"] = backward_mems
        file = (
            f"pgo_{cfg.solver_device}_{cfg.inner_optim.solver}_{cfg.num_poses}_"
            f"{cfg.dataset_size}_{cfg.batch_size}.mat"
        )
        savemat(file, results)

    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


@hydra.main(config_path="./configs/", config_name="pose_graph")
def main(cfg):
    log.info((subprocess.check_output("lscpu", shell=True).strip()).decode())

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # create (or load) dataset
    rng = torch.Generator()
    rng.manual_seed(0)
    dtype = torch.float64
    pg, _ = theg.PoseGraphDataset.generate_synthetic_3D(
        num_poses=cfg.num_poses,
        rotation_noise=cfg.rotation_noise,
        translation_noise=cfg.translation_noise,
        loop_closure_ratio=cfg.loop_closure_ratio,
        loop_closure_outlier_ratio=cfg.loop_closure_outlier_ratio,
        batch_size=cfg.batch_size,
        dataset_size=cfg.dataset_size,
        generator=rng,
        dtype=dtype,
    )

    results_path = pathlib.Path(os.getcwd())
    run(cfg, pg, results_path)


if __name__ == "__main__":
    main()
