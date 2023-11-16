# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# SE(3) convention in this example is translation then rotation

import logging
import random
import subprocess
from typing import Dict, List, Type, Union, cast

import hydra
import numpy as np
import omegaconf
import torch
from scipy.io import savemat

import theseus as th
import theseus.utils.examples as theg
from theseus.utils import Profiler, Timer

# Logger
log = logging.getLogger(__name__)


def print_histogram(
    pg: theg.PoseGraphDataset, var_dict: Dict[str, torch.Tensor], msg: str
):
    log.info(msg)
    with torch.no_grad():
        poses = [th.SE3(tensor=var_dict[pose.name]) for pose in pg.poses]
        histogram = theg.pg_histogram(poses=poses, edges=pg.edges)
    for line in histogram.split("\n"):
        log.info(line)


def get_batch_data(
    pg_batch: theg.PoseGraphDataset, pose_indices: List[int], gt_pose_indices: List[int]
):
    batch = {
        pg_batch.poses[index].name: pg_batch.poses[index].tensor
        for index in pose_indices
    }
    batch.update({pg_batch.poses[0].name + "__PRIOR": pg_batch.poses[0].tensor.clone()})
    batch.update(
        {
            pg_batch.gt_poses[index].name: pg_batch.gt_poses[index].tensor
            for index in gt_pose_indices
        }
    )
    batch.update(
        {edge.relative_pose.name: edge.relative_pose.tensor for edge in pg_batch.edges}
    )
    return batch


def pose_loss(
    pose_vars: Union[List[th.SE2], List[th.SE3]],
    gt_pose_vars: Union[List[th.SE2], List[th.SE3]],
) -> torch.Tensor:
    loss: torch.Tensor = torch.zeros(
        1, dtype=pose_vars[0].dtype, device=pose_vars[0].device
    )
    poses_batch = th.SE3(tensor=torch.cat([pose.tensor for pose in pose_vars]))
    gt_poses_batch = th.SE3(
        tensor=torch.cat([gt_pose.tensor for gt_pose in gt_pose_vars])
    )
    pose_loss = th.local(poses_batch, gt_poses_batch).norm(dim=1)
    loss += pose_loss.sum()
    return loss


def _maybe_reset_cuda_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _maybe_get_cuda_max_mem_alloc():
    return (
        torch.cuda.max_memory_allocated() / 1048576
        if torch.cuda.is_available()
        else torch.nan
    )


def run(cfg: omegaconf.OmegaConf):
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
        translation_noise=cfg.translation_noise,
        rotation_noise=cfg.rotation_noise,
        loop_closure_ratio=cfg.loop_closure_ratio,
        loop_closure_outlier_ratio=cfg.loop_closure_outlier_ratio,
        batch_size=cfg.batch_size,
        dataset_size=cfg.dataset_size,
        generator=rng,
        dtype=dtype,
    )

    device = torch.device(cfg.device)
    dtype = torch.float64
    profiler = Profiler(cfg.profile)
    pg.to(device=device)

    with torch.no_grad():
        pose_loss_ref = pose_loss(pg.poses, pg.gt_poses).item()
    log.info(f"POSE LOSS (no learning):  {pose_loss_ref: .3f}")

    # Create the objective
    pg_batch = pg.get_batch_dataset(0)

    log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=dtype)
    robust_loss_cls = th.WelschLoss

    objective = th.Objective(dtype=torch.float64)

    pose_indices: List[int] = [index for index, _ in enumerate(pg_batch.poses)]
    gt_pose_indices: List[int] = []

    for edge in pg_batch.edges:
        relative_pose_cost = th.Between(
            pg_batch.poses[edge.i],
            pg_batch.poses[edge.j],
            edge.relative_pose,
            edge.weight,
        )
        robust_relative_pose_cost = th.RobustCostFunction(
            cost_function=relative_pose_cost,
            loss_cls=robust_loss_cls,
            log_loss_radius=log_loss_radius,
        )
        objective.add(robust_relative_pose_cost)

    if cfg.inner_optim.regularize:
        pose_prior_cost = th.Difference(
            var=pg_batch.poses[0],
            target=pg_batch.poses[0].copy(new_name=pg_batch.poses[0].name + "__PRIOR"),
            cost_weight=th.ScaleCostWeight(
                torch.tensor(cfg.inner_optim.reg_w, dtype=dtype)
            ),
        )
        objective.add(pose_prior_cost)

    if cfg.inner_optim.ratio_known_poses > 0.0:
        pose_prior_weight = th.ScaleCostWeight(100 * torch.ones(1, dtype=dtype))
        for i in range(len(pg_batch.poses)):
            if np.random.rand() > cfg.inner_optim.ratio_known_poses:
                continue
            objective.add(
                th.Difference(
                    pg_batch.poses[i],
                    pg_batch.gt_poses[i],
                    pose_prior_weight,
                    name=f"pose_diff_{i}",
                )
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
        objective.to(device),
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
        linear_solver_cls=getattr(th, cfg.inner_optim.linear_solver_cls),
    )

    # Set up Theseus layer
    theseus_optim = th.TheseusLayer(
        optimizer,
        vectorize=cfg.inner_optim.vectorize,
        empty_cuda_cache=cfg.inner_optim.empty_cuda_cache,
    )
    theseus_optim.to(device=device)

    # Outer optimization loop
    log_loss_radius_tensor = torch.nn.Parameter(
        torch.tensor([[3.0]], device=device, dtype=dtype)
    )
    model_optimizer = torch.optim.Adam([log_loss_radius_tensor], lr=cfg.outer_optim.lr)

    num_epochs = cfg.outer_optim.num_epochs

    def run_batch(batch_idx: int):
        log.info(f" ------------------- Batch {batch_idx} ------------------- ")

        pg_batch = pg.get_batch_dataset(batch_idx=batch_idx)
        theseus_inputs = get_batch_data(pg_batch, pose_indices, gt_pose_indices)
        theseus_inputs["log_loss_radius"] = log_loss_radius_tensor.clone()

        with torch.no_grad():
            pose_loss_ref = pose_loss(pg_batch.poses, pg_batch.gt_poses)

        timer = Timer(device)
        with timer:
            _maybe_reset_cuda_peak_mem()
            profiler.enable()
            theseus_outputs, _ = theseus_optim.forward(
                input_tensors=theseus_inputs,
                optimizer_kwargs={**cfg.inner_optim.optimizer_kwargs},
            )
            profiler.disable()
        forward_time = 1000 * timer.elapsed_time
        forward_mem = _maybe_get_cuda_max_mem_alloc()
        log.info(f"Forward pass took {forward_time} ms.")
        log.info(f"Forward pass used {forward_mem} GPU MBs.")

        with timer:
            _maybe_reset_cuda_peak_mem()
            profiler.enable()
            model_optimizer.zero_grad()
            loss = (
                pose_loss(pose_vars, pg_batch.gt_poses) - pose_loss_ref
            ) / pose_loss_ref
            loss.backward()
            model_optimizer.step()
            backward_mem = _maybe_get_cuda_max_mem_alloc()
            profiler.disable()
        backward_time = 1000 * timer.elapsed_time
        log.info(f"Backward pass took {backward_time} ms.")
        log.info(f"Backward pass used {backward_mem} GPU MBs.")

        loss_value = torch.sum(loss.detach()).item()
        log.info(
            f"Loss: {loss_value} "
            f"Kernel Radius: exp({log_loss_radius_tensor.data.item()})="
            f"{torch.exp(log_loss_radius_tensor.data).item()}"
        )

        print_histogram(pg_batch, theseus_outputs, "Output histogram:")

        return [forward_time, backward_time, forward_mem, backward_mem, loss.item()]

    forward_times = []
    backward_times = []
    forward_mems = []
    backward_mems = []
    losses = []
    for epoch in range(num_epochs):
        log.info(f" ******************* EPOCH {epoch} ******************* ")

        forward_time_epoch = []
        backward_time_epoch = []
        forward_mem_epoch = []
        backward_mem_epoch = []
        losses_epoch = []
        for batch_idx in range(pg.num_batches):
            if batch_idx == cfg.outer_optim.max_num_batches:
                break
            forward_time, backward_time, forward_mem, backward_mem, loss = run_batch(
                batch_idx
            )

            forward_time_epoch.append(forward_time)
            backward_time_epoch.append(backward_time)
            forward_mem_epoch.append(forward_mem)
            backward_mem_epoch.append(backward_mem)
            losses_epoch.append(loss)

        forward_times.append(forward_time_epoch)
        backward_times.append(backward_time_epoch)
        forward_mems.append(forward_mem_epoch)
        backward_mems.append(backward_mem_epoch)
        losses.append(losses_epoch)

        results = omegaconf.OmegaConf.to_container(cfg)
        results["forward_time"] = forward_times
        results["backward_time"] = backward_times
        results["forward_mem"] = forward_mems
        results["backward_mem"] = backward_mems
        fname = (
            f"pgo_{cfg.device}_{cfg.inner_optim.linear_solver_cls.lower()}_{cfg.num_poses}_"
            f"{cfg.dataset_size}_{cfg.batch_size}.mat"
        )
        print(fname)
        if cfg.savemat:
            savemat(fname, results)

    profiler.print()
    return losses


@hydra.main(
    config_path="../configs/pose_graph",
    config_name="pose_graph_synthetic",
    version_base="1.1",
)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
