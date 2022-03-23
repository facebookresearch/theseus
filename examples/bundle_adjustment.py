# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, Type

import hydra
import numpy as np
import omegaconf
import torch

import theseus as th
import theseus.utils.examples as theg


BACKWARD_MODE = {
    "implicit": th.BackwardMode.IMPLICIT,
    "full": th.BackwardMode.FULL,
    "truncated": th.BackwardMode.TRUNCATED,
}

# Smaller values} result in error
th.SO3.SO3_EPS = 1e-6


def print_histogram(
    ba: theg.BundleAdjustmentDataset, var_dict: Dict[str, torch.Tensor], msg: str
):
    print(msg)
    theg.ba_histogram(
        cameras=[
            theg.Camera(
                th.SE3(data=var_dict[c.pose.name]),
                c.focal_length,
                c.calib_k1,
                c.calib_k2,
            )
            for c in ba.cameras
        ],
        points=[th.Point3(data=var_dict[pt.name]) for pt in ba.points],
        observations=ba.observations,
    )


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


def run(cfg: omegaconf.OmegaConf):
    # create (or load) dataset
    ba = theg.BundleAdjustmentDataset.generate_synthetic(
        num_cameras=cfg.num_cameras,
        num_points=cfg.num_points,
        average_track_length=cfg.average_track_length,
        track_locality=cfg.track_locality,
    )

    # hyper parameters (ie outer loop's parameters)
    log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=torch.float64)

    # Set up objective
    objective = th.Objective(dtype=torch.float64)

    for obs in ba.observations:
        cam = ba.cameras[obs.camera_index]
        cost_function = theg.Reprojection(
            camera_pose=cam.pose,
            world_point=ba.points[obs.point_index],
            focal_length=cam.focal_length,
            calib_k1=cam.calib_k1,
            calib_k2=cam.calib_k2,
            log_loss_radius=log_loss_radius,
            image_feature_point=obs.image_feature_point,
        )
        objective.add(cost_function)
    dtype = objective.dtype

    # Add regularization
    if cfg.inner_optim.regularize:
        zero_point3 = th.Point3(dtype=dtype, name="zero_point")
        identity_se3 = th.SE3(dtype=dtype, name="zero_se3")
        w = np.sqrt(cfg.inner_optim.reg_w)
        damping_weight = th.ScaleCostWeight(w * torch.ones(1, dtype=dtype))
        for name, var in objective.optim_vars.items():
            target: th.LieGroup
            if isinstance(var, th.SE3):
                target = identity_se3
            elif isinstance(var, th.Point3):
                target = zero_point3
            else:
                assert False
            objective.add(
                th.eb.VariableDifference(
                    var, damping_weight, target, name=f"reg_{name}"
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
    orig_poses = {cam.pose.name: cam.pose.data.clone() for cam in ba.cameras}
    orig_points = {pt.name: pt.data.clone() for pt in ba.points}

    # Outer optimization loop
    loss_radius_tensor = torch.nn.Parameter(torch.tensor([-1], dtype=torch.float64))
    model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=cfg.outer_optim.lr)

    num_epochs = cfg.outer_optim.num_epochs
    camera_pose_vars = [
        theseus_optim.objective.optim_vars[c.pose.name] for c in ba.cameras
    ]

    theseus_inputs = get_batch(ba, orig_poses, orig_points)
    print_histogram(ba, theseus_inputs, "Input histogram:")
    for epoch in range(num_epochs):
        print(f" ******************* EPOCH {epoch} ******************* ")
        model_optimizer.zero_grad()
        theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

        theseus_outputs, info = theseus_optim.forward(
            input_data=theseus_inputs,
            optimizer_kwargs={
                "verbose": cfg.inner_optim.verbose,
                "track_err_history": cfg.inner_optim.track_err_history,
                "backward_mode": BACKWARD_MODE[cfg.inner_optim.backward_mode],
            },
        )
        print_histogram(ba, theseus_outputs, "Output histogram:")

        loss: torch.Tensor = 0  # type:ignore
        for i in range(len(ba.cameras)):
            loss += th.local(camera_pose_vars[i], ba.gt_cameras[i].pose).norm(dim=1)
        loss.backward()
        model_optimizer.step()
        loss_value = torch.sum(loss.detach()).item()

        print(
            f"Epoch: {epoch} Loss: {loss_value} "
            f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
            f"{torch.exp(loss_radius_tensor.data).item()}"
        )


@hydra.main(config_path="./configs/", config_name="bundle_adjustment")
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
