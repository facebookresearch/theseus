# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pathlib
import random
import time
from typing import Dict, List, Type

import hydra
import numpy as np
import omegaconf
import torch

import theseus as th
import theseus.utils.examples as theg

# Logger
log = logging.getLogger(__name__)


def print_histogram(
    ba: theg.BundleAdjustmentDataset, var_dict: Dict[str, torch.Tensor], msg: str
):
    log.info(msg)
    histogram = theg.ba_histogram(
        cameras=[
            theg.Camera(
                th.SE3(tensor=var_dict[c.pose.name]),
                c.focal_length,
                c.calib_k1,
                c.calib_k2,
            )
            for c in ba.cameras
        ],
        points=[th.Point3(tensor=var_dict[pt.name]) for pt in ba.points],
        observations=ba.observations,
    )
    for line in histogram.split("\n"):
        log.info(line)


# loads (the only) batch
def get_batch(
    ba: theg.BundleAdjustmentDataset,
    orig_poses: Dict[str, torch.Tensor],
    orig_points: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    retv = {}
    for cam in ba.cameras:
        retv[cam.pose.name] = orig_poses[cam.pose.name].clone()
    for pt in ba.points:
        retv[pt.name] = orig_points[pt.name].clone()
    return retv


def save_epoch(
    results_path: pathlib.Path,
    epoch: int,
    log_loss_radius: th.Vector,
    theseus_outputs: Dict[str, torch.Tensor],
    info: th.optimizer.OptimizerInfo,
    loss_value: float,
    total_time: float,
):
    def _clone(t_):
        return t_.detach().cpu().clone()

    results = {
        "log_loss_radius": _clone(log_loss_radius.tensor),
        "theseus_outputs": dict((s, _clone(t)) for s, t in theseus_outputs.items()),
        "err_history": info.err_history,  # type: ignore
        "loss": loss_value,
        "total_time": total_time,
    }
    torch.save(results, results_path / f"results_epoch{epoch}.pt")


def camera_loss(
    ba: theg.BundleAdjustmentDataset, camera_pose_vars: List[th.LieGroup]
) -> torch.Tensor:
    loss: torch.Tensor = 0  # type:ignore
    for i in range(len(ba.cameras)):
        camera_loss = th.local(camera_pose_vars[i], ba.gt_cameras[i].pose).norm(dim=1)
        loss += camera_loss
    return loss


def run(cfg: omegaconf.OmegaConf, results_path: pathlib.Path):
    # create (or load) dataset
    ba = theg.BundleAdjustmentDataset.generate_synthetic(
        num_cameras=cfg.num_cameras,
        num_points=cfg.num_points,
        average_track_length=cfg.average_track_length,
        track_locality=cfg.track_locality,
        feat_random=1.5,
        outlier_feat_random=70,
    )
    ba.save_to_file(results_path / "ba.txt", gt_path=results_path / "ba_gt.txt")

    # hyper parameters (ie outer loop's parameters)
    log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=torch.float64)

    # Set up objective
    objective = th.Objective(dtype=torch.float64)

    weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=ba.cameras[0].pose.dtype))
    for obs in ba.observations:
        cam = ba.cameras[obs.camera_index]
        cost_function = th.eb.Reprojection(
            camera_pose=cam.pose,
            world_point=ba.points[obs.point_index],
            focal_length=cam.focal_length,
            calib_k1=cam.calib_k1,
            calib_k2=cam.calib_k2,
            image_feature_point=obs.image_feature_point,
            weight=weight,
        )
        robust_cost_function = th.RobustCostFunction(
            cost_function,
            th.HuberLoss,
            log_loss_radius,
            name=f"robust_{cost_function.name}",
        )
        objective.add(robust_cost_function)
    dtype = objective.dtype

    # Add regularization
    if cfg.inner_optim.regularize:
        zero_point3 = th.Point3(dtype=dtype, name="zero_point")
        identity_se3 = th.SE3(dtype=dtype, name="zero_se3")
        w = np.sqrt(cfg.inner_optim.reg_w)
        damping_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype))
        for name, var in objective.optim_vars.items():
            target: th.Manifold
            if isinstance(var, th.SE3):
                target = identity_se3
            elif isinstance(var, th.Point3):
                target = zero_point3
            else:
                assert False
            objective.add(
                th.Difference(var, target, damping_weight, name=f"reg_{name}")
            )

    camera_pose_vars: List[th.LieGroup] = [
        objective.optim_vars[c.pose.name] for c in ba.cameras  # type: ignore
    ]
    if cfg.inner_optim.ratio_known_cameras > 0.0:
        w = 100.0
        camera_weight = th.ScaleCostWeight(100 * torch.ones(1, dtype=dtype))
        for i in range(len(ba.cameras)):
            if np.random.rand() > cfg.inner_optim.ratio_known_cameras:
                continue
            objective.add(
                th.Difference(
                    camera_pose_vars[i],
                    ba.gt_cameras[i].pose,
                    camera_weight,
                    name=f"camera_diff_{i}",
                )
            )

    # Create optimizer
    optimizer_cls: Type[th.NonlinearLeastSquares] = getattr(
        th, cfg.inner_optim.optimizer_cls
    )
    optimizer = optimizer_cls(
        objective,
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
    )

    # Set up Theseus layer
    theseus_optim = th.TheseusLayer(optimizer)

    # copy the poses/pts to feed them to each outer iteration
    orig_poses = {cam.pose.name: cam.pose.tensor.clone() for cam in ba.cameras}
    orig_points = {pt.name: pt.tensor.clone() for pt in ba.points}

    # Outer optimization loop
    loss_radius_tensor = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=cfg.outer_optim.lr)

    num_epochs = cfg.outer_optim.num_epochs

    theseus_inputs = get_batch(ba, orig_poses, orig_points)
    theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

    with torch.no_grad():
        camera_loss_ref = camera_loss(ba, camera_pose_vars).item()
    log.info(f"CAMERA LOSS (no learning):  {camera_loss_ref: .3f}")
    print_histogram(ba, theseus_inputs, "Input histogram:")
    for epoch in range(num_epochs):
        log.info(f" ******************* EPOCH {epoch} ******************* ")
        start_time = time.time_ns()
        model_optimizer.zero_grad()
        theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

        theseus_outputs, info = theseus_optim.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs={
                "verbose": cfg.inner_optim.verbose,
                "track_err_history": cfg.inner_optim.track_err_history,
                "backward_mode": cfg.inner_optim.backward_mode,
                "__keep_final_step_size__": cfg.inner_optim.keep_step_size,
            },
        )

        loss = (camera_loss(ba, camera_pose_vars) - camera_loss_ref) / camera_loss_ref
        loss.backward()
        model_optimizer.step()
        loss_value = torch.sum(loss.detach()).item()
        end_time = time.time_ns()

        print_histogram(ba, theseus_outputs, "Output histogram:")
        log.info(
            f"Epoch: {epoch} Loss: {loss_value} "
            f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
            f"{torch.exp(loss_radius_tensor.data).item()}"
        )
        log.info(f"Epoch took {(end_time - start_time) / 1e9: .3f} seconds")

        save_epoch(
            results_path,
            epoch,
            log_loss_radius,
            theseus_outputs,
            info,
            loss_value,
            end_time - start_time,
        )


@hydra.main(config_path="./configs/", config_name="bundle_adjustment")
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    results_path = pathlib.Path(os.getcwd())
    run(cfg, results_path)


if __name__ == "__main__":
    main()
